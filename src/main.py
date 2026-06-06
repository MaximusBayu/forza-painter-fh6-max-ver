# Discord: A-Dawg#0001 (AE)
# Supports: Forza Horizon 5 / Forza Horizon 6 profile probing
# Officially: MS Store/XBOX PC App, Steam.
# Unofficially: Every version that isn't running on a console of via cloud gaming should work.
# License: MIT
# Year: 2022

import sys
import argparse
import importlib
import ctypes, sys
import psutil
import ctypes
import struct
import subprocess
#from geometrize.geometrize import geometrize_image
#from geometrize.internal_classes import ThreadManager
from native import *
from internal_classes import *
from game_profiles import iter_profiles, PROFILES
from geometry_json import RECTANGLE, TRIANGLE, ROTATED_ELLIPSE, load_normalized_geometry
import colorsys
import os

from utils import load_cv2, parse_int

FH6_DISCOVERED_TABLE_POINTER_DELTA = 0x1E
FH6_CIRCLE_BASE_SIZE = 63.0
FH6_RECTANGLE_BASE_SIZE = 127.0
# S3 triangle: these two constants are the ONLY pieces that need in-game
# discovery (Tools -> Inspect table on a placed triangle vinyl). The shape-id
# byte is unknown until then; while it is None the importer safely SKIPS
# triangles instead of writing a guessed byte into game memory. Override via
# the FORZA_PAINTER_TRIANGLE_SHAPE_ID / _BASE_SIZE env vars once discovered.
FH6_TRIANGLE_SHAPE_ID = parse_int(os.environ.get("FORZA_PAINTER_TRIANGLE_SHAPE_ID"))
try:
    FH6_TRIANGLE_BASE_SIZE = float(os.environ.get("FORZA_PAINTER_TRIANGLE_BASE_SIZE", FH6_RECTANGLE_BASE_SIZE))
except (TypeError, ValueError):
    FH6_TRIANGLE_BASE_SIZE = FH6_RECTANGLE_BASE_SIZE


def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def show_image(image):
    print("External preview windows are disabled. Use the desktop app preview panel instead.")

def get_pid(game_key=None, pid_override=None):
    if pid_override:
        try:
            proc = psutil.Process(pid_override)
            proc_name = proc.name()
            for profile in iter_profiles(game_key):
                if proc_name.lower() in [name.lower() for name in profile.process_names]:
                    print("{} detected as {} (pid {})".format(profile.label, proc_name, pid_override))
                    return pid_override, profile
            if game_key:
                profile = next(iter_profiles(game_key))
                print("{} selected for {} (pid {})".format(profile.label, proc_name, pid_override))
                return pid_override, profile
            print("PID {} is running as {}, but it does not match a supported profile.".format(pid_override, proc_name))
        except psutil.Error as exc:
            print("Unable to inspect pid {}: {}".format(pid_override, exc))
        return -1, None

    process_lookup = {}
    for proc in psutil.process_iter():
        try:
            process_lookup[proc.name().lower()] = proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    for profile in iter_profiles(game_key):
        for process_name in profile.process_names:
            pid = process_lookup.get(process_name.lower())
            if pid:
                print("{} detected as {} (pid {})".format(profile.label, process_name, pid))
                return pid, profile
    if game_key:
        profile_names = ", ".join(next(iter_profiles(game_key)).process_names)
        print("{} is not running ({})".format(game_key.upper(), profile_names))
    else:
        names = ", ".join(name for profile in iter_profiles() for name in profile.process_names)
        print("No supported Forza Horizon process is running ({})".format(names))
    return -1, None

def find_livery_signature(pid, base_addr, profile):
    for start_offset, block_size in profile.scan_regions:
        start_address = base_addr + start_offset
        for signature in profile.signature_patterns:
            print("Scanning {} Base+{:x}..Base+{:x}".format(
                profile.label, start_offset, start_offset + block_size))
            relative_address = scan_block(pid, start_address, block_size, signature)
            if relative_address != -1:
                return start_address + relative_address, signature
    return -1, None

def calculate_CLivery(pid, profile):
    base_addr = get_base_address(pid)
    print("Attempting to scan for {} livery address:".format(profile.label))
    preAddrA, signature = find_livery_signature(pid, base_addr, profile)
    if preAddrA == -1:
        print("Unsupported {} version and cannot find a matching pattern.".format(profile.label))
        print("FH6 support may need a new signature or offsets for this game build.")
        return -1
    print("Signature {} found at Base+{:x}".format(signature.hex(" "), preAddrA - base_addr))
    if read_long(pid, preAddrA) != read_long(pid, preAddrA + profile.validation_mirror_offset):
        print("Matching signature failed validation at Base+{:x}".format(preAddrA - base_addr))
        return -1
    addrA = dereference_pointer(pid, preAddrA + profile.livery_root_pointer_offset)
    print("Found livery root pointer at Base+{0:x}".format(preAddrA + profile.livery_root_pointer_offset - base_addr))
    addrB = dereference_pointer(pid, addrA + profile.editor_pointer_offset)
    if addrB == 0:
        print("Create Vinyl Group menu not detected")
        return -1
    cLivery = dereference_pointer(pid, addrB + profile.livery_pointer_offset)
    if cLivery == 0:
        print("Create Vinyl Group menu not detected")
        return -1
    return cLivery

def diagnose_livery(pid, profile):
    try:
        base_addr = get_base_address(pid)
    except Exception as exc:
        print("Unable to open process {}. Try running as administrator. {}".format(pid, exc))
        return False

    print("{} diagnostics for pid {}".format(profile.label, pid))
    print("Base address: 0x{:x}".format(base_addr))
    found_valid = False
    for start_offset, block_size in profile.scan_regions:
        start_address = base_addr + start_offset
        for signature in profile.signature_patterns:
            relative_address = scan_block(pid, start_address, block_size, signature)
            if relative_address == -1:
                print("No match in Base+{:x}..Base+{:x}".format(start_offset, start_offset + block_size))
                continue
            absolute_address = start_address + relative_address
            print("Signature {} found at Base+{:x}".format(signature.hex(" "), absolute_address - base_addr))
            try:
                mirror_equal = read_long(pid, absolute_address) == read_long(pid, absolute_address + profile.validation_mirror_offset)
                root = dereference_pointer(pid, absolute_address + profile.livery_root_pointer_offset)
                editor = dereference_pointer(pid, root + profile.editor_pointer_offset) if root else 0
                livery = dereference_pointer(pid, editor + profile.livery_pointer_offset) if editor else 0
                print("Validation mirror: {}".format("OK" if mirror_equal else "FAILED"))
                print("Root pointer: 0x{:x}".format(root))
                print("Editor pointer: 0x{:x}".format(editor))
                print("Livery pointer: 0x{:x}".format(livery))
                found_valid = found_valid or (mirror_equal and root != 0 and editor != 0 and livery != 0)
            except Exception as exc:
                print("Pointer chain failed: {}".format(exc))
    if not found_valid:
        print("No validated livery pointer chain found.")
    return found_valid

def draw_memory_shape(pid: int, profile, shape: Shape, index: int, cLiveryLayerTable: int, liveryCount: int):
    if index >= liveryCount:
        return True
    current_layer_address = dereference_pointer(pid, cLiveryLayerTable + (index * 0x8))
    pos_data = struct.pack('f', shape.x) + struct.pack('f', -shape.y)
    try:
        write_process_memory(pid, current_layer_address + profile.layer_position_offset, pos_data)
        if shape.type_id == ROTATED_ELLIPSE:
            scale_divisor = FH6_CIRCLE_BASE_SIZE
        elif shape.type_id == TRIANGLE:
            scale_divisor = FH6_TRIANGLE_BASE_SIZE
        else:
            scale_divisor = FH6_RECTANGLE_BASE_SIZE
        scale_data = struct.pack('f', shape.w / scale_divisor) + struct.pack('f', shape.h / scale_divisor)
        write_process_memory(pid, current_layer_address + profile.layer_scale_offset, scale_data)
        rot_data = struct.pack('f', 360 - shape.rot_deg)
        write_process_memory(pid, current_layer_address + profile.layer_rotation_offset, rot_data)
        color_data = shape.color.get_struct()
        write_process_memory(pid, current_layer_address + profile.layer_color_offset, color_data)
        if shape.type_id == ROTATED_ELLIPSE:
            shape_id_data = struct.pack('B', 102)
            write_process_memory(pid, current_layer_address + profile.layer_shape_id_offset, shape_id_data)
        elif shape.type_id == TRIANGLE and FH6_TRIANGLE_SHAPE_ID is not None:
            shape_id_data = struct.pack('B', int(FH6_TRIANGLE_SHAPE_ID) & 0xFF)
            write_process_memory(pid, current_layer_address + profile.layer_shape_id_offset, shape_id_data)
        elif shape.type_id == RECTANGLE:
            shape_id_data = struct.pack('B', 101)
            write_process_memory(pid, current_layer_address + profile.layer_shape_id_offset, shape_id_data)
        mask_flag = struct.pack('B', 1 if shape.is_mask else 0)
        write_process_memory(pid, current_layer_address + profile.layer_mask_offset, mask_flag)
    except:
        if index > 0:
            print("Detected grouped vinyl in slot " + str(index+1))
        print("ERROR: You probably forgot to ungroup one of your vinyls.")
        print("Also ensure you are in the Vinyl Group Editor, not applying the vinyl or a livery to the car.")
        return False
    return True
    

def load_geometry(
    path,
    game_key=None,
    preview_enabled=True,
    pid_override=None,
    layer_count_address=None,
    layer_table_address=None,
    layer_count_value=None,
):
    try:
        data = load_normalized_geometry(path)
    except Exception as exc:
        print("Not a valid generated geometry .json file: {}".format(exc))
        return

    # validation and build our collection of shapes
    image_w, image_h = data['shapes'][0]['data'][2:]
    bg_r, bg_g, bg_b, bg_a = data['shapes'][0]['color']
    shapes = []
    triangle_skipped = 0
    
    # If the exported geometry has a visible rectangle background, add it.
    # Transparent PNG exports often include an alpha=0 background rectangle;
    # writing that layer can turn into a visible fallback color in-game.
    if bg_a > 0:
        shapes.append(Shape(1, int(image_w//2), int(image_h//2), image_w, image_h, 0, Color(bg_r,bg_g,bg_b,bg_a), False))

    for shape in data['shapes'][1:]:
        #shape.color = [r,g,b,a]
        #shape.data = [x,y,w,h,rot_deg]
        if len(shape.get('color', [])) == 4 and int(shape['color'][3]) <= 0:
            continue
        if shape['type'] == ROTATED_ELLIPSE:
            x,y,w,h,rot_deg = shape['data']
            r,g,b,a = shape['color']
            shapes.append(Shape(shape['type'], x, y, w, h, rot_deg, Color(r,g,b,a), False))
        elif shape['type'] == RECTANGLE:
            data = shape['data']
            x, y, w, h = data[:4]
            # S1: carry rectangle rotation (5th value) when present. The write
            # path already stores a rotation field for every layer.
            rot_deg = data[4] if len(data) >= 5 else 0
            r,g,b,a = shape['color']
            shapes.append(Shape(shape['type'], x, y, w, h, rot_deg, Color(r,g,b,a), False))
        elif shape['type'] == TRIANGLE:
            # S3: triangles only import once the FH6 shape-id byte is known.
            # Until then skip them rather than write a guessed byte to memory.
            if FH6_TRIANGLE_SHAPE_ID is None:
                triangle_skipped += 1
                continue
            data = shape['data']
            x, y, w, h = data[:4]
            rot_deg = data[4] if len(data) >= 5 else 0
            r,g,b,a = shape['color']
            shapes.append(Shape(shape['type'], x, y, w, h, rot_deg, Color(r,g,b,a), False))
        else:
            print("Skipping unsupported shape type {}.".format(shape.get("type")))
    if triangle_skipped:
        print(
            "Skipped {} triangle shape(s): the FH6 triangle shape-id is not configured. "
            "Discover it in-game (Tools -> Inspect table) and set FORZA_PAINTER_TRIANGLE_SHAPE_ID."
            .format(triangle_skipped)
        )
    if len(shapes) == 0:
        print("No shapes were loaded. Check your exported geometry .json")
        return
    
    loaded = load_cv2()
    if loaded:
        cv2, np = loaded
        preview = np.zeros((int(image_h), int(image_w), 3), np.uint8)
        if bg_a > 0:
            preview = cv2.rectangle(preview, (0,0), (int(image_w), int(image_h)), (bg_b, bg_g, bg_r), thickness=-1)
        else:
            preview[:, :] = (38, 38, 38)
        for shape in shapes:
            if shape.color.a <= 0:
                continue
            if shape.type_id == ROTATED_ELLIPSE:
                preview = cv2.ellipse(preview, (shape.x, shape.y), (int(shape.h), int(shape.w)), -90 + shape.rot_deg, 0., 360, (shape.color.b, shape.color.g, shape.color.r), thickness=-1)
            elif shape.type_id == RECTANGLE:
                rot_deg = getattr(shape, "rot_deg", 0) or 0
                if rot_deg:
                    theta = np.deg2rad(float(rot_deg))
                    ct, st = np.cos(theta), np.sin(theta)
                    hw, hh = shape.w / 2.0, shape.h / 2.0
                    corners = []
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        dx, dy = sx * hw, sy * hh
                        corners.append([shape.x + dx * ct - dy * st, shape.y + dx * st + dy * ct])
                    pts = np.array(corners, dtype=np.int32)
                    preview = cv2.fillConvexPoly(preview, pts, (shape.color.b, shape.color.g, shape.color.r))
                else:
                    x0 = int(round(shape.x - shape.w / 2))
                    y0 = int(round(shape.y - shape.h / 2))
                    x1 = int(round(shape.x + shape.w / 2))
                    y1 = int(round(shape.y + shape.h / 2))
                    preview = cv2.rectangle(preview, (x0, y0), (x1, y1), (shape.color.b, shape.color.g, shape.color.r), thickness=-1)
            elif shape.type_id == TRIANGLE:
                theta = np.deg2rad(float(getattr(shape, "rot_deg", 0) or 0))
                ct, st = np.cos(theta), np.sin(theta)
                hw, hh = shape.w / 2.0, shape.h / 2.0
                corners = [[shape.x + dx * ct - dy * st, shape.y + dx * st + dy * ct]
                           for dx, dy in ((0.0, -hh), (-hw, hh), (hw, hh))]
                pts = np.array(corners, dtype=np.int32)
                preview = cv2.fillConvexPoly(preview, pts, (shape.color.b, shape.color.g, shape.color.r))

        if preview_enabled:
            print("Here is a preview of your image, click it then press any key to start!")
            show_image(preview)
            cv2.imwrite("preview.png", preview)
    elif preview_enabled:
        print("Preview unavailable because OpenCV/Numpy could not be loaded. Import will continue.")
    
    # Finding the game PID
    pid, profile = get_pid(game_key, pid_override)
    if pid == -1:
        return

    if layer_count_address:
        if layer_count_value:
            current_livery_count = int(layer_count_value)
            print("Manual layer count address 0x{0:x}; using template layer count {1}".format(layer_count_address, current_livery_count))
        elif game_key == "fh6":
            raw_count = read_process_memory(pid, layer_count_address, 2)
            current_livery_count = int.from_bytes(raw_count, byteorder=sys.byteorder) if len(raw_count) == 2 else 0
            print("Manual FH6 layer count address 0x{0:x} -> {1}".format(layer_count_address, current_livery_count))
        else:
            current_livery_count = read_int(pid, layer_count_address)
            print("Manual layer count address 0x{0:x} -> {1}".format(layer_count_address, current_livery_count))
        if not layer_table_address:
            table_pointer_field = layer_count_address + FH6_DISCOVERED_TABLE_POINTER_DELTA
            layer_table_address = dereference_pointer(pid, table_pointer_field)
            print("Manual table pointer field 0x{0:x} -> 0x{1:x}".format(table_pointer_field, layer_table_address))
        cLiveryLayerTable = layer_table_address
    else:
        # Calculate the pointer chain to the cLiveryLayerTable
        cLivery = calculate_CLivery(pid, profile)
        if cLivery == -1:
            return
        print("CLivery found at {0:x}".format(cLivery))
        cLiveryGroup = dereference_pointer(pid, cLivery + profile.livery_group_offset)
        if cLiveryGroup == 0:
            print("cLiveryGroup is invalid...")
            print("You are probably not in `Create Vinyl Group` menu...")
            return
        print("CLiveryGroup found at {0:x}".format(cLiveryGroup))
        current_livery_count = read_int(pid, cLiveryGroup + profile.livery_count_offset)
        cLiveryLayerTable = dereference_pointer(pid, cLiveryGroup + profile.layer_table_offset)

    # If we have less than 100 shapes, user has likely made a mistake
    if current_livery_count < 100:
        print("READ THE INSTRUCTIONS")
        print("You must load a vinyl group (ALL SPHERES) with your desired shape count (minimum 100) first!")
        print("500, 1000, 1500, 2000 or 3000 is recommended")
        print("Make sure to ungroup the vinyl before starting 1also!")
        return

    if cLiveryLayerTable == 0:
        print("cLiveryLayer table is invalid...")
        print("You are probably not in `Create Vinyl Group` menu..")
        return
    print("CLiveryLayer table found at {0:x}".format(cLiveryLayerTable))

    # FH recalculates the saved vinyl/group bounds from mask layers. Without
    # these boundary masks the design can look correct in the editor but save
    # with a blank cover, paste blank onto the car, or recover only after the
    # user manually moves the group.
    boundary_masks = [
        Shape(1, -int(image_w//4), int(image_h//2), float(image_w//2), float(image_h*1.5), 0, Color(0,0,0,255), True),
        Shape(1, image_w + int(image_w//4), int(image_h//2), float(image_w//2), float(image_h*1.5), 0, Color(0,0,0,255), True),
        Shape(1, int(image_w//2), -int(image_h//4), float(image_w + image_w), float(image_h//2), 0, Color(0,0,0,255), True),
        Shape(1, int(image_w//2), image_h + int(image_h//4), float(image_w + image_w), float(image_h//2), 0, Color(0,0,0,255), True),
    ]
    reserved_mask_layers = len(boundary_masks)
    drawable_capacity = max(0, min(int(current_livery_count), 3000) - reserved_mask_layers)

    if len(shapes) > drawable_capacity:
        print(
            "Geometry has {} drawable layers but FH bounds reserve {} layers; trimming to {} drawable layers.".format(
                len(shapes), reserved_mask_layers, drawable_capacity
            )
        )
        shapes = shapes[:drawable_capacity]

    shapes.extend(boundary_masks)

    print(
        "Drawable layers to import: {} + {} FH bounds layers / template layers: {}".format(
            max(0, len(shapes) - reserved_mask_layers),
            reserved_mask_layers,
            current_livery_count,
        )
    )
    
    # Enumerate every template slot. Any unused slot is hidden so larger templates
    # do not leave their original spheres visible after importing smaller JSON.
    clear_shape = Shape(1, 0, 0, 0, 0, 0, Color(0, 0, 0, 0), False)
    for i in range(current_livery_count):
        shape = shapes[i] if i < len(shapes) else clear_shape
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == current_livery_count:
            print("Writing layer {}/{}".format(i + 1, current_livery_count), flush=True)
        if not draw_memory_shape(pid, profile, shape, i, cLiveryLayerTable, current_livery_count):
            return
    
    print("DONE!")

    # Show the background color as the ideal car color in HSV format
    h,s,v = colorsys.rgb_to_hsv(bg_r / float(255), bg_g / float(255), bg_b / float(255))
    print("The ideal background color for the car is:\n{:.2f},{:.2f},{:.2f}".format(h,s,v))

def parse_args(args):
    parser = argparse.ArgumentParser(description="Import generated geometry into Forza Horizon vinyl editor.")
    parser.add_argument("--game", choices=PROFILES.keys(), default=os.environ.get("FORZA_PAINTER_GAME"),
                        help="Target game profile. Defaults to auto-detecting a running supported game.")
    parser.add_argument("--pid", type=int, default=None, help="Use a specific running game process id.")
    parser.add_argument("--no-preview", action="store_true", help="Skip the OpenCV preview prompt.")
    parser.add_argument("--diagnose", action="store_true", help="Run read-only process signature diagnostics.")
    parser.add_argument("--layer-count-address", type=parse_int, default=None,
                        help="Manual live layer-count address, e.g. 0x16debce3a9a. Bypasses signature chain.")
    parser.add_argument("--layer-table-address", type=parse_int, default=None,
                        help="Manual live layer-table address. If omitted with --layer-count-address, uses count+0x1e pointer field.")
    parser.add_argument("--layer-count-value", type=int, default=None,
                        help="Known template layer count. Used by FH6 because the live count field is u16 inside a larger structure.")
    parser.add_argument("geometry_path", nargs="*", help="Generated .json geometry file path.")
    parsed = parser.parse_args(args[1:])
    parsed.geometry_path = " ".join(parsed.geometry_path)
    return parsed

def main(args):
    if not is_64bit():
        print("Your Python version is 32-bit. Please install 64-bit Python.\nThis is required for IPC with Forza Horizon as it is a 64-bit process.")
        return
    parsed = parse_args(args)
    if parsed.diagnose:
        pid, profile = get_pid(parsed.game, parsed.pid)
        if pid != -1:
            diagnose_livery(pid, profile)
        return
    if not parsed.geometry_path:
        print("You must drag in a generated geometry .json file!")
        return
    path = parsed.geometry_path

    if not os.path.isfile(path):
        print("{} is not a valid file path!".format(path))
        return
    ext = path.split('.')[-1].lower()
    #accepted_image_formats = ["jpg", "jpeg", "png", "bmp"]
    is_geometry = ext == "json"
    if not is_geometry:# and not ext in accepted_image_formats:
        print("Expected 1 file as the only argument.")
        print("An image file, or an generated .json geometry file.")
        return
    if is_geometry:
        load_geometry(
            path,
            parsed.game,
            not parsed.no_preview,
            parsed.pid,
            parsed.layer_count_address,
            parsed.layer_table_address,
            parsed.layer_count_value,
        )
    # else:
    #     geometrize_image(path)

if __name__ == "__main__":
    if is_admin() or os.environ.get("FORZA_PAINTER_NO_ELEVATE") == "1":
        # Capture any exceptions
        try:
            main(sys.argv)
        except BaseException:
            print(sys.exc_info()[0])
            import traceback
            print(traceback.format_exc())
        finally:
            #ThreadManager.ensure_all_threads_killed()
            if os.environ.get("FORZA_PAINTER_NO_PAUSE") != "1":
                print("Press Enter to continue ...")
                input()
    else:
        # Run as admin
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, subprocess.list2cmdline(sys.argv), None, 1)
