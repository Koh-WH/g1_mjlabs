import joblib
import numpy as np
import sys
import os
import argparse

def convert_pkl_to_csv(pkl_file):
    if not pkl_file:
        print("❌ Error: No file provided.")
        return

    print(f"📂 Opening {pkl_file}...")
    
    if not os.path.exists(pkl_file):
        print(f"❌ Error: File not found: {pkl_file}")
        return

    # Define Output Directory
    output_dir = "/home/koh-wh/Downloads/mjlab/src/mjlab/pkl/csv"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load the file
        data = joblib.load(pkl_file)
        
        # 2. Extract Data
        if isinstance(data, dict):
            first_key = list(data.keys())[0]
            if isinstance(data[first_key], dict) and "root_trans_offset" in data[first_key]:
                print(f"   -> Found nested entry: '{first_key}'")
                inner_data = data[first_key]
            else:
                inner_data = data
        else:
            print("❌ Error: Pickle file structure is not a dictionary.")
            return

        # 3. Get Components
        if "root_trans_offset" in inner_data:
            root_pos = inner_data["root_trans_offset"]
        else:
            print("❌ Missing 'root_trans_offset'")
            return

        if "root_rot" in inner_data:
            root_rot = inner_data["root_rot"]
        else:
            print("❌ Missing 'root_rot'")
            return

        if "dof" in inner_data:
            dof_pos = inner_data["dof"]
        else:
             print("❌ Missing 'dof' (joint angles)")
             return

        # ======================================================================
        # 🧪 SMART PADDING LOGIC (23 -> 29 Joints)
        # ======================================================================
        n_frames, n_joints = dof_pos.shape
        target_joints = 29
        
        if n_joints == 29:
            print("   ✅ Format matches (29 joints). No padding needed.")
            final_dof = dof_pos
            
        elif n_joints == 23:
            print("   ⚠️  Detected 23 joints (Missing Wrists). Injecting zeros...")
            # Create a container for 29 joints
            final_dof = np.zeros((n_frames, 29), dtype=dof_pos.dtype)
            
            # 1. Copy Legs, Waist, Left Shoulder, Left Elbow (Indices 0-18)
            final_dof[:, :19] = dof_pos[:, :19]
            
            # 2. Inject 0 for Left Wrist (Indices 19, 20, 21) -> Already 0
            
            # 3. Copy Right Shoulder, Right Elbow (Old Indices 19-23 -> New 22-26)
            final_dof[:, 22:26] = dof_pos[:, 19:]
            
            # 4. Inject 0 for Right Wrist (Indices 26, 27, 28) -> Already 0
            
        elif n_joints < 29:
            print(f"   ⚠️  Detected {n_joints} joints. Appending zeros to reach 29.")
            # Generic padding at the end (fallback for other weird formats)
            padding = np.zeros((n_frames, 29 - n_joints))
            final_dof = np.hstack([dof_pos, padding])
            
        else:
            # If > 29, we just crop it? Or leave it? usually safe to leave or crop.
            print(f"   ⚠️  Detected {n_joints} joints. Cropping to 29.")
            final_dof = dof_pos[:, :29]

        # ======================================================================

        # 4. Stack & Save
        full_motion = np.hstack([root_pos, root_rot, final_dof])
        
        # Output Filename
        base_name = os.path.basename(pkl_file)
        name_no_ext = os.path.splitext(base_name)[0]
        output_filename = os.path.join(output_dir, name_no_ext + ".csv")

        np.savetxt(output_filename, full_motion, delimiter=",")
        print(f"✅ Success! Saved to: {output_filename}")
        print(f"   Original Shape: {dof_pos.shape} -> New Shape: {final_dof.shape}")

    except Exception as e:
        print(f"❌ Failed to process file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PKL motion files to CSV for Mjlab.")
    parser.add_argument("input_file", nargs="?", help="Path to the .pkl file")
    parser.add_argument("--file", help="Path to the .pkl file (optional flag)")
    args = parser.parse_args()

    target_file = args.file if args.file else args.input_file
    if target_file:
        convert_pkl_to_csv(target_file)
    else:
        parser.print_help()