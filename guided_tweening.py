bl_info = {
    "name": "Guided Inbetween",
    "author": "Paxon",
    "version": (1, 0, 1),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Guided Tween",
    "description": "Stoke order & Frame-aware guided inbetween/tweening with marker/arrow customization, stroke-end advance, integrity checks.",
    "category": "Animation",
}

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector
from bpy.props import (
    BoolProperty, FloatVectorProperty, IntProperty, FloatProperty, EnumProperty)
import traceback, time, math

# -------------------------
# Globals / State
# -------------------------
_draw_handle = None
_timer_keep = False

_guide_strokes_world = []
_guide_index = 0
_monitored_frame = None
_last_current_frame_count = 0

# Pending state for advance-on-release
_pending_active = False
_pending_target = 0
_pending_time = 0.0
_pending_last_pointlen = 0
_debounce_seconds = 0.35

# integrity re-check counter
_advances_since_check = 0

# shader
_SHADER = gpu.shader.from_builtin("UNIFORM_COLOR")

# -------------------------
# Scene properties
# -------------------------
def register_scene_props():
    # marker settings
    bpy.types.Scene.gi_marker_shape = EnumProperty(
        name="Marker Shape",
        items=[('SQUARE','Square',''), ('TRIANGLE','Triangle',''), ('CIRCLE','Circle','')],
        default='SQUARE')
    bpy.types.Scene.gi_marker_size = IntProperty(
        name="Marker Size", default=8, min=4, max=48)
    bpy.types.Scene.gi_start_color = FloatVectorProperty(
        name="Start Color", subtype="COLOR", size=4,
        default=(0.2,1.0,0.2,1.0), min=0.0, max=1.0)
    bpy.types.Scene.gi_end_color = FloatVectorProperty(
        name="End Color", subtype="COLOR", size=4,
        default=(1.0,0.2,0.2,1.0), min=0.0, max=1.0)

    # arrow settings
    bpy.types.Scene.gi_show_arrows = BoolProperty(name="Show Arrows", default=True)
    bpy.types.Scene.gi_arrow_shape = EnumProperty(
        name="Arrow Shape",
        items=[
            ('TRI','Triangle','Simple triangular arrow tip'),
            ('RECT','Rectangle','Small rectangle arrow'),
            ('DOTTED','Dotted','Series of small rectangles (dotted)'),
            ('RECT_TRI','Arrow','Rectangle shaft with triangular tip')
        ], default='TRI')
    bpy.types.Scene.gi_arrow_color = FloatVectorProperty(
        name="Arrow Color", subtype="COLOR", size=4,
        default=(1.0,0.9,0.2,1.0), min=0.0, max=1.0)
    bpy.types.Scene.gi_arrow_density = IntProperty(
        name="Arrow Density", default=12, min=1, max=200,
        description="Higher → more arrows along stroke")
    bpy.types.Scene.gi_arrow_size = FloatProperty(
        name="Arrow Size", default=12.0, min=2.0, max=64.0)
    bpy.types.Scene.gi_arrow_margin = IntProperty(
        name="Arrow Margin %", default=12, min=0, max=40,
        description="Percent of stroke ends to skip when placing arrows")

    # behavior / misc
    bpy.types.Scene.gi_lock_frame = BoolProperty(name="Lock Monitored Frame", default=False)
    bpy.types.Scene.gi_advance_on_release = BoolProperty(
        name="Advance on Stroke End", default=False,
        description="If on: wait until a stroke finishes before advancing the hint")
    bpy.types.Scene.gi_integrity_check = IntProperty(
        name="Integrity Check Interval", default=3, min=1, max=20,
        description="Perform re-check of source strokes every N advances (background)")

def unregister_scene_props():
    for p in ("gi_marker_shape","gi_marker_size","gi_start_color","gi_end_color",
              "gi_show_arrows","gi_arrow_shape","gi_arrow_color","gi_arrow_density","gi_arrow_size","gi_arrow_margin",
              "gi_lock_frame","gi_advance_on_release","gi_integrity_check"):
        try:
            delattr(bpy.types.Scene, p)
        except Exception:
            pass

# -------------------------
# Helpers
# -------------------------
def debug(s):
    print("[GuidedInbetween] " + str(s))

def get_active_gp(context):
    o = context.object
    if o and getattr(o,"type",None) == "GREASEPENCIL":
        return o
    return None

def find_nearest_previous_frame(layer, current_frame_num):
    nearest = None
    best = float("inf")
    for fr in layer.frames:
        if fr.frame_number < current_frame_num and getattr(fr,"drawing",None):
            if not fr.drawing.strokes:
                continue
            d = current_frame_num - fr.frame_number
            if d < best:
                nearest = fr; best = d
    return nearest

def get_previous_frame_strokes_world(context):
    obj = get_active_gp(context)
    if not obj: return []
    if not obj.data.layers or obj.data.layers.active is None: return []
    layer = obj.data.layers.active
    current = context.scene.frame_current
    nearest = find_nearest_previous_frame(layer, current)
    if not nearest: return []
    mat = obj.matrix_world
    strokes = []
    try:
        for stroke in nearest.drawing.strokes:
            pts = [mat @ Vector(p.position) for p in stroke.points]
            if len(pts) >= 1:
                strokes.append(pts)
    except Exception:
        debug("copy error:\n"+traceback.format_exc())
        return []
    return strokes

def get_frame_stroke_count(context, frame_number):
    obj = get_active_gp(context)
    if not obj: return 0
    if not obj.data.layers or obj.data.layers.active is None: return 0
    layer = obj.data.layers.active
    fr = None
    if getattr(layer,"active_frame",None) and layer.active_frame.frame_number==frame_number:
        fr = layer.active_frame
    else:
        for f in layer.frames:
            if f.frame_number == frame_number:
                fr = f; break
    if not fr or not getattr(fr,"drawing",None): return 0
    try:
        return len(fr.drawing.strokes)
    except Exception:
        try: return sum(1 for _ in fr.drawing.strokes)
        except Exception: return 0

def get_last_stroke_pointlen(context, frame_number):
    """Return number of points in the last stroke on frame (or 0)."""
    try:
        obj = get_active_gp(context)
        if not obj: return 0
        layer = obj.data.layers.active
        if not layer: return 0
        # find frame
        fr = None
        if getattr(layer, "active_frame", None) and layer.active_frame.frame_number == frame_number:
            fr = layer.active_frame
        else:
            for f in layer.frames:
                if f.frame_number == frame_number:
                    fr = f; break
        if not fr or not getattr(fr, "drawing", None):
            return 0
        if not fr.drawing.strokes:
            return 0
        last = fr.drawing.strokes[-1]
        return len(last.points)
    except Exception:
        return 0

# -------------------------
# Drawing primitives (2D)
# -------------------------
def draw_point(pos, color):
    _SHADER.bind()
    _SHADER.uniform_float("color", tuple(color))
    batch = batch_for_shader(_SHADER, "POINTS", {"pos": [pos]})
    batch.draw(_SHADER)

def draw_line_strip(pts, color, lw=2.0):
    _SHADER.bind()
    _SHADER.uniform_float("color", tuple(color))
    try: gpu.state.line_width_set(lw)
    except Exception: pass
    batch = batch_for_shader(_SHADER, "LINE_STRIP", {"pos": pts})
    batch.draw(_SHADER)

def draw_lines_flat(pairs, color, lw=2.0):
    _SHADER.bind()
    _SHADER.uniform_float("color", tuple(color))
    try: gpu.state.line_width_set(lw)
    except Exception: pass
    batch = batch_for_shader(_SHADER, "LINES", {"pos": pairs})
    batch.draw(_SHADER)

def draw_tris_flat(tris, color):
    _SHADER.bind()
    _SHADER.uniform_float("color", tuple(color))
    batch = batch_for_shader(_SHADER, "TRIS", {"pos": tris})
    batch.draw(_SHADER)

# shapes
def square_around(center, size):
    x,y = center
    hs = size/2.0
    return [(x-hs,y-hs),(x+hs,y-hs),(x+hs,y+hs),(x-hs,y+hs)]

def triangle_around(center, size):
    x,y = center
    h = size*0.86
    return [(x,y+h/2.0),(x-h/2.0,y-h/2.0),(x+h/2.0,y-h/2.0)]

def circle_fan(center, size, segments=12):
    x,y = center
    pts = []
    for i in range(segments):
        a0 = 2*math.pi*(i/segments)
        a1 = 2*math.pi*((i+1)/segments)
        pts.extend([(x,y),(x+math.cos(a0)*size,y+math.sin(a0)*size),(x+math.cos(a1)*size,y+math.sin(a1)*size)])
    return pts

def draw_marker_shape(center, shape, size, color):
    if shape == 'SQUARE':
        poly = square_around(center, size)
        tris = [poly[0],poly[1],poly[2], poly[0],poly[2],poly[3]]
        draw_tris_flat(tris, color)
    elif shape == 'TRIANGLE':
        tri = triangle_around(center, size)
        tris = [tri[0],tri[1],tri[2]]
        draw_tris_flat(tris, color)
    else:
        tris = circle_fan(center, size/1.5, segments=14)
        draw_tris_flat(tris, color)

# arrow primitives
def make_arrow_tri(mid, direction, size):
    d = direction.normalized()
    perp = Vector((-d.y, d.x))
    tip = (mid + d*size).to_tuple()
    bl = (mid - d*(size*0.33) + perp*(size*0.33)).to_tuple()
    br = (mid - d*(size*0.33) - perp*(size*0.33)).to_tuple()
    return [tip, bl, br]

def make_rect(mid, direction, size, width_factor=0.35):
    d = direction.normalized()
    perp = Vector((-d.y, d.x))
    half_len = size*0.6
    half_w = size*width_factor
    a = (mid - d*half_len + perp*half_w).to_tuple()
    b = (mid + d*half_len + perp*half_w).to_tuple()
    c = (mid + d*half_len - perp*half_w).to_tuple()
    dpt = (mid - d*half_len - perp*half_w).to_tuple()
    return [a,b,c, a,c,dpt]

# -------------------------
# Draw handler (main)
# -------------------------
def draw_guides():
    try:
        if not _guide_strokes_world: return
        context = bpy.context
        region = context.region
        rv3d = getattr(context,"region_data",None)
        if region is None or rv3d is None: return

        scene = context.scene
        start_col = getattr(scene,"gi_start_color",(0.2,1.0,0.2,1.0))
        end_col = getattr(scene,"gi_end_color",(1.0,0.2,0.2,1.0))
        show_arrows = getattr(scene,"gi_show_arrows",True)
        arrow_col = getattr(scene,"gi_arrow_color",(1.0,0.9,0.2,1.0))
        arrow_shape = getattr(scene,"gi_arrow_shape","TRI")
        density = getattr(scene,"gi_arrow_density",12)
        arrow_size = getattr(scene,"gi_arrow_size",12.0)
        margin_pct = getattr(scene,"gi_arrow_margin",12)
        mark_shape = getattr(scene,"gi_marker_shape","SQUARE")
        mark_size = getattr(scene,"gi_marker_size",8)

        idx = max(0, min(_guide_index, len(_guide_strokes_world)-1))
        stroke_world = _guide_strokes_world[idx]

        pts2d = []
        for w in stroke_world:
            p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, w)
            if p2 is not None:
                pts2d.append((p2.x, p2.y))
        if not pts2d: return

        gpu.state.blend_set("ALPHA")
        try: gpu.state.point_size_set(6)
        except Exception: pass

        # single-point stroke marker
        if len(pts2d) == 1:
            draw_marker_shape(pts2d[0], mark_shape, mark_size, start_col)
            gpu.state.blend_set("NONE")
            return

        # stroke path
        draw_line_strip(pts2d, (0.28,0.68,1.0,0.32), lw=2.0)

        # start/end markers
        draw_marker_shape(pts2d[0], mark_shape, mark_size, start_col)
        draw_marker_shape(pts2d[-1], mark_shape, mark_size, end_col)

        # arrows/ticks - compute segments to place arrows skipping margin
        seg_count = len(pts2d)-1
        margin_steps = int(math.ceil(seg_count * (margin_pct/100.0)))
        margin_steps = max(0, min(margin_steps, seg_count//2))

        step = max(1, int(seg_count / max(1, density)))

        if show_arrows:
            tris = []
            for i in range(margin_steps, seg_count - margin_steps, step):
                a = Vector(pts2d[i]); b = Vector(pts2d[i+1]); seg = b - a
                if seg.length < 1e-6: continue
                mid = (a + b) / 2.0
                if arrow_shape == 'TRI':
                    tri = make_arrow_tri(mid, seg, arrow_size)
                    tris.extend(tri)
                elif arrow_shape == 'RECT':
                    rect = make_rect(mid, seg, arrow_size, width_factor=0.28)
                    tris.extend(rect)
                elif arrow_shape == 'DOTTED':
                    rect = make_rect(mid, seg, arrow_size*0.6, width_factor=0.25)
                    tris.extend(rect)
                elif arrow_shape == 'RECT_TRI':
                    center = mid - seg.normalized()*(arrow_size*0.25)
                    rect = make_rect(center, seg, arrow_size*0.9, width_factor=0.22)
                    tris.extend(rect)
                    tri = make_arrow_tri(mid + seg.normalized()*(arrow_size*0.5), seg, arrow_size*0.9)
                    tris.extend(tri)
            if tris:
                draw_tris_flat(tris, arrow_col)
        else:
            mids = []
            for i in range(0, seg_count, step):
                a = Vector(pts2d[i]); b = Vector(pts2d[i+1])
                mids.append(((a+b)/2.0).to_tuple())
            if mids:
                draw_point(mids[0], (1.0,1.0,0.0,0.88))
                _SHADER.uniform_float("color", (1.0,1.0,0.0,0.88))
                batch = batch_for_shader(_SHADER, "POINTS", {"pos": mids})
                batch.draw(_SHADER)

        gpu.state.blend_set("NONE")

    except Exception:
        debug("Exception in draw_guides:\n"+traceback.format_exc())

# -------------------------
# Timer / watcher (debounce + last-stroke-pointlen check)
# -------------------------
def _watch_timer():
    global _last_current_frame_count, _guide_index, _timer_keep, _draw_handle
    global _monitored_frame, _pending_active, _pending_target, _pending_time, _pending_last_pointlen, _advances_since_check

    if not _timer_keep:
        return None
    if _draw_handle is None:
        _timer_keep = False
        return None

    try:
        context = bpy.context
        scene = context.scene
        current_frame = context.scene.frame_current
        current_count = get_frame_stroke_count(context, current_frame)

        # FRAME CHANGE HANDLING
        if _monitored_frame is None or current_frame != _monitored_frame:
            if getattr(scene,"gi_lock_frame", False) and _monitored_frame is not None:
                _last_current_frame_count = current_count
                _pending_active = False
            else:
                _monitored_frame = current_frame
                _last_current_frame_count = current_count
                if _guide_strokes_world:
                    _guide_index = min(_last_current_frame_count, max(0, len(_guide_strokes_world)-1))
                else:
                    _guide_index = 0
                _pending_active = False
                for w in bpy.context.window_manager.windows:
                    for a in w.screen.areas:
                        if a.type == "VIEW_3D": a.tag_redraw()

        else:
            # If a pending commit is active: check stability (both stroke count and last stroke pointlen)
            if _pending_active:
                # if count changed, update target and reset timer & last_pointlen
                if current_count != _pending_target:
                    _pending_target = current_count
                    _pending_last_pointlen = get_last_stroke_pointlen(context, _monitored_frame)
                    _pending_time = time.time()
                else:
                    # check last stroke pointlen
                    cur_len = get_last_stroke_pointlen(context, _monitored_frame)
                    now = time.time()
                    if cur_len != _pending_last_pointlen:
                        # still growing — update and reset timer
                        _pending_last_pointlen = cur_len
                        _pending_time = now
                    else:
                        # stable: check debounce
                        if (now - _pending_time) >= _debounce_seconds:
                            added = _pending_target - _last_current_frame_count
                            if added > 0:
                                _guide_index = min(_guide_index + added, max(0, len(_guide_strokes_world)-1))
                                _last_current_frame_count = _pending_target
                                _advances_since_check += added
                                if _advances_since_check >= getattr(scene, "gi_integrity_check", 3):
                                    _advances_since_check = 0
                                    _guide_strokes_world[:] = get_previous_frame_strokes_world(context)
                                for w in bpy.context.window_manager.windows:
                                    for a in w.screen.areas:
                                        if a.type == "VIEW_3D": a.tag_redraw()
                            _pending_active = False

            else:
                # No pending active: normal detection
                if current_count > _last_current_frame_count:
                    added = current_count - _last_current_frame_count
                    if getattr(scene, "gi_advance_on_release", False):
                        # start pending and record target/time and last stroke pointlen
                        _pending_active = True
                        _pending_target = current_count
                        _pending_time = time.time()
                        _pending_last_pointlen = get_last_stroke_pointlen(context, _monitored_frame)
                    else:
                        # immediate advance
                        _guide_index = min(_guide_index + added, max(0, len(_guide_strokes_world)-1))
                        _last_current_frame_count = current_count
                        _advances_since_check += added
                        if _advances_since_check >= getattr(scene, "gi_integrity_check", 3):
                            _advances_since_check = 0
                            _guide_strokes_world[:] = get_previous_frame_strokes_world(context)
                        for w in bpy.context.window_manager.windows:
                            for a in w.screen.areas:
                                if a.type == "VIEW_3D": a.tag_redraw()

                elif current_count < _last_current_frame_count:
                    # stroke removal
                    _last_current_frame_count = current_count
                    _pending_active = False
                    if _guide_strokes_world:
                        _guide_index = min(current_count, max(0, len(_guide_strokes_world)-1))
                    else:
                        _guide_index = 0
                    for w in bpy.context.window_manager.windows:
                        for a in w.screen.areas:
                            if a.type == "VIEW_3D": a.tag_redraw()

    except Exception:
        debug("Exception in _watch_timer:\n"+traceback.format_exc())

    return 0.12

# -------------------------
# Operators
# -------------------------
class GP_OT_GuideStart(bpy.types.Operator):
    bl_idname = "gp_guide.start"
    bl_label = "Start Guides"

    def execute(self, context):
        global _draw_handle, _guide_strokes_world, _guide_index, _last_current_frame_count, _timer_keep, _monitored_frame, _pending_active, _advances_since_check
        obj = get_active_gp(context)
        if not obj:
            self.report({"WARNING"},"Select a Grease Pencil object first")
            return {"CANCELLED"}
        strokes = get_previous_frame_strokes_world(context)
        if not strokes:
            self.report({"WARNING"},"No strokes found in previous frame")
            return {"CANCELLED"}
        _guide_strokes_world = strokes
        _monitored_frame = context.scene.frame_current
        _last_current_frame_count = get_frame_stroke_count(context, _monitored_frame)
        _guide_index = min(_last_current_frame_count, max(0, len(_guide_strokes_world)-1))
        _pending_active = False
        _advances_since_check = 0
        if _draw_handle is None:
            try:
                _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_guides, (), "WINDOW", "POST_PIXEL")
            except Exception:
                debug("handler add failed:\n"+traceback.format_exc())
                self.report({"ERROR"},"Failed to add handler (see console)")
                _draw_handle = None
                return {"CANCELLED"}
        _timer_keep = True
        bpy.app.timers.register(_watch_timer)
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == "VIEW_3D": a.tag_redraw()
        self.report({"INFO"}, f"Guides started — {len(_guide_strokes_world)} hints loaded; monitoring frame {_monitored_frame}")
        return {"FINISHED"}

class GP_OT_GuideStop(bpy.types.Operator):
    bl_idname = "gp_guide.stop"
    bl_label = "Stop Guides"
    def execute(self, context):
        global _draw_handle, _guide_strokes_world, _timer_keep, _monitored_frame, _pending_active
        if _draw_handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_draw_handle,"WINDOW")
            except Exception:
                debug("handler remove failed:\n"+traceback.format_exc())
            _draw_handle = None
        _timer_keep = False
        _guide_strokes_world = []
        _monitored_frame = None
        _pending_active = False
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == "VIEW_3D": a.tag_redraw()
        self.report({"INFO"},"Guides stopped")
        return {"FINISHED"}

# -------------------------
# UI Panel
# -------------------------
class GP_PT_GuidedPanelFull(bpy.types.Panel):
    bl_label = "Guided Inbetween"
    bl_idname = "GP_PT_guided_final_panel_v2"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Guided Tween"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.operator("gp_guide.start", icon="PLAY")
        layout.operator("gp_guide.stop", icon="CANCEL")
        layout.separator()

        box = layout.box()
        box.label(text="Marker Settings")
        row = box.row(align=True)
        row.prop(scene, "gi_marker_shape", text="Shape")
        row.prop(scene, "gi_marker_size", text="Size")
        row = box.row(align=True)
        row.prop(scene, "gi_start_color", text="Start")
        row.prop(scene, "gi_end_color", text="End")

        box2 = layout.box()
        box2.label(text="Arrow Settings")
        box2.prop(scene, "gi_show_arrows", text="Show Arrows")
        if scene.gi_show_arrows:
            box2.prop(scene, "gi_arrow_shape", text="Shape")
            box2.prop(scene, "gi_arrow_color", text="Color")
            box2.prop(scene, "gi_arrow_density", text="Density")
            box2.prop(scene, "gi_arrow_size", text="Size")
            box2.prop(scene, "gi_arrow_margin", text="Margin %")

        box3 = layout.box()
        box3.label(text="Behavior")
        box3.prop(scene, "gi_lock_frame", text="Lock Monitored Frame")
        box3.prop(scene, "gi_advance_on_release", text="Advance on Stroke End")
        box3.prop(scene, "gi_integrity_check", text="Integrity Check Interval")


# -------------------------
# Register
# -------------------------
classes = (GP_OT_GuideStart, GP_OT_GuideStop, GP_PT_GuidedPanelFull)

def register():
    register_scene_props()
    for c in classes: bpy.utils.register_class(c)
    debug("Guided Inbetween v2 registered")

def unregister():
    global _draw_handle, _timer_keep
    if _draw_handle is not None:
        try: bpy.types.SpaceView3D.draw_handler_remove(_draw_handle,"WINDOW")
        except Exception: pass
        _draw_handle = None
    _timer_keep = False
    for c in reversed(classes):
        try: bpy.utils.unregister_class(c)
        except Exception: pass
    unregister_scene_props()
    debug("Guided Inbetween v2 unregistered")

if __name__ == "__main__":
    register()