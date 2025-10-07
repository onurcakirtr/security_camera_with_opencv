#cli.py
import argparse
import logging
import sys
import cv2

from motion import detect_motion
from face import train_model, load_model, recognize_faces
from recorder import Recorder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Security Cam with Multi-Face Recognition"
    )
    parser.add_argument('-train', action='store_true',
                        help="Train face recognition model")
    parser.add_argument('-cam', action='store_true',
                        help="Start camera monitoring")
    parser.add_argument('-gui', action='store_true',
                        help="Launch the GUI for configuring options")
    parser.add_argument('--cam-num', type=int, default=1,
                        help="Number of cameras to use (default: 1)")
    parser.add_argument('--min-area', type=int, default=500,
                        help="Minimum contour area for motion detection")
    parser.add_argument('--face-interval', type=int, default=10,
                        help="Run face recognition every N frames")
    parser.add_argument('--duration', type=int, default=20,
                        help="Max recording duration (seconds)")
    parser.add_argument('--snapshot', action='store_true',
                        help="Enable snapshots during recording")
    parser.add_argument('--snapshot-interval', type=int, default=5,
                        help="Snapshot interval (seconds)")
    parser.add_argument('--no-record', action='store_true',
                        help="Detect motion but do not record video")
    parser.add_argument('--no-display', action='store_true',
                        help="Do not show the video window")
    parser.add_argument('--resolution', type=str, default='640x480',
                        help="Camera resolution, e.g. 640x480 or 320x240")
    parser.add_argument('--fps', type=float, default=20.0,
                        help="Frame rate for recording")
    parser.add_argument('--model', type=str, default='model.pkl',
                        help="Path to face model file")
    parser.add_argument('--data', type=str, default='faces',
                        help="Directory of face images for training")
    parser.add_argument('--threshold', type=float, default=0.7,
                        help="Confidence threshold for unknown faces (0–1)")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable debug logging")
    return parser.parse_args()


def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        level=level)
    
    if args.gui:
        from gui import launch_gui
        launch_gui()
        return

    if args.train:
        train_model(data_dir=args.data, model_path=args.model)
        return

    # Load or train face model
    try:
        clf, le = load_model(args.model)
    except FileNotFoundError:
        logging.info("Model not found; training now.")
        train_model(data_dir=args.data, model_path=args.model)
        clf, le = load_model(args.model)
    trusted_set = set(le.classes_)

    # Haar Cascade
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if cascade.empty():
        logging.error("Failed to load Haar cascade")
        sys.exit(1)

    # Initialize cameras based on --cam-num
    cameras = []
    for cam_id in range(args.cam_num):
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.resolution:
            w, h = map(int, args.resolution.split('x'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            logging.warning(f"Camera {cam_id} could not be opened.")
            cameras.append(None)
        else:
            recorder = Recorder(
                output_dir=f'recordings/cam{cam_id}',
                fps=args.fps,
                duration=args.duration,
                snapshot=args.snapshot,
                snapshot_interval=args.snapshot_interval,
                no_record=args.no_record
            )
            cameras.append({
                'id': cam_id,
                'cap': cap,
                'avg': None,
                'recorder': recorder,
                'frame_no': 0,
                'last_ann': [],
                'motion': False,
                'roi': None,
                'trusted_present': False
            })

    while True:
        frames = []
        any_motion = False
        trusted_found = False

        # Process each camera
        for cam in cameras:
            if not cam:
                frames.append(None)
                continue

            ret, frame = cam['cap'].read()
            if not ret:
                frames.append(None)
                continue

            cam['frame_no'] += 1
            cam['avg'], cam['motion'], cam['roi'] = detect_motion(frame, cam['avg'], args.min_area)

            # Face recognition at intervals
            if cam['frame_no'] % args.face_interval == 0:
                cam['last_ann'] = recognize_faces(
                    frame, clf, le, trusted_set, threshold=args.threshold
                )

            # Check for trusted faces
            cam['trusted_present'] = any(
                name in trusted_set for (_, _, _, _, name, _) in cam['last_ann']
            )

            if cam['motion']:
                any_motion = True
            if cam['trusted_present']:
                trusted_found = True

            frames.append(frame)

        # Decide recording per camera
        for cam, frame in zip(cameras, frames):
            if not cam or frame is None:
                continue

            record = False
            # No trusted face & any motion -> record
            if not cam['trusted_present'] and any_motion:
                record = True
            # No trusted anywhere & any motion -> record all
            elif any_motion and not trusted_found:
                record = True

            if record and cam['recorder'].writer is None and not args.no_record:
                cam['recorder'].start(frame)
            if cam['recorder'].writer:
                cam['recorder'].update(frame)

            # Visualization
            if cam['roi']:
                x, y, w, h = cam['roi']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (x1, y1, x2, y2, name, color) in cam['last_ann']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display combined
        if not args.no_display:
            valid = [f for f in frames if f is not None]
            if valid:
                if len(valid) > 1:
                    disp = cv2.hconcat(valid)
                else:
                    disp = valid[0]
                cv2.imshow("Multi-Cam Security", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    for cam in cameras:
        if cam:
            cam['cap'].release()
            if cam['recorder'].writer:
                cam['recorder'].stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
