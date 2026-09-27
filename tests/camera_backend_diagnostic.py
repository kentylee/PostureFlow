import cv2

backends = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("DEFAULT", 0),
    ("MSMF", cv2.CAP_MSMF),
]

for index in range(5):
    for backend_name, backend in backends:
        print(f"Testing camera index {index} with backend {backend_name}...")

        if backend == 0:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            print(f"Could not open index {index} with {backend_name}")
            cap.release()
            continue

        success, frame = cap.read()

        if success and frame is not None:
            print(f"SUCCESS: camera index {index}, backend {backend_name}")

            while True:
                cv2.imshow(f"Camera {index} - {backend_name}", frame)
                success, frame = cap.read()

                if not success or frame is None:
                    print("Frame failed while previewing.")
                    break

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                if key == ord("n"):
                    break
        else:
            print(f"Opened but could not read frame: index {index}, backend {backend_name}")

        cap.release()
        cv2.destroyAllWindows()

print("Finished camera test.")