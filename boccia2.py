import cv2
import numpy as np
import math

class FieldMeasure:
    # ---------- settings ----------
    PREVIEW_SCALE = 0.5     # main preview size
    LOUPE_SCALE   = 2.0     # magnification inside loupe
    LOUPE_WIN     = 200     # loupe window side (px)
    FIELD_W_CM    = 400.0   # set real width here
    FIELD_H_CM    = 500.0   # set real height here
    LOUPE_BORDER  = 2       # Border thickness for loupe
    LOUPE_BORDER_COLOR = (0, 255, 0) # Green border
    TARGET_COLOR = (0, 0, 255)      # Red color for the target dot
    TARGET_RADIUS = 2
    # --------------------------------

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(img_path)

        self.preview = cv2.resize(
            self.img, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
            interpolation=cv2.INTER_AREA)

        self.field_pts, self.ball_pt1 = [], None
        self.cursor_preview = (0, 0) # Cursor coordinates on the preview image
        self.loupe_window_created = False

    # ----- mouse callback -----
    def _mouse(self, e, x, y, flags, _):
        self.cursor_preview = (x, y)
        if e == cv2.EVENT_LBUTTONDOWN:
            X = int(x / self.PREVIEW_SCALE)
            Y = int(y / self.PREVIEW_SCALE)
            if len(self.field_pts) < 4:
                self.field_pts.append((X, Y))
                print(f"Corner {len(self.field_pts)} → ({X}, {Y})")
            elif self.ball_pt1 is None:
                self.ball_pt1 = (X, Y)
                print(f"Ball 1 → ({X}, {Y})")
        elif e == cv2.EVENT_MOUSEMOVE:
            pass

    # ----- create loupe image -----
    def _loupe_image(self):
        x_p, y_p = self.cursor_preview
        X = int(x_p / self.PREVIEW_SCALE)
        Y = int(y_p / self.PREVIEW_SCALE)

        half = int(self.LOUPE_WIN / (2 * self.LOUPE_SCALE))
        x1, y1 = max(0, X - half), max(0, Y - half)
        x2, y2 = min(self.img.shape[1], X + half), min(self.img.shape[0], Y + half)
        patch = self.img[y1:y2, x1:x2]

        loupe = cv2.resize(patch, (self.LOUPE_WIN, self.LOUPE_WIN),
                            interpolation=cv2.INTER_NEAREST)

        # Draw a red dot at the center of the zoomed window
        center = self.LOUPE_WIN // 2
        cv2.circle(loupe, (center, center), self.TARGET_RADIUS, self.TARGET_COLOR, -1)
        cv2.rectangle(loupe, (0, 0), (self.LOUPE_WIN - 1, self.LOUPE_WIN - 1), self.LOUPE_BORDER_COLOR, self.LOUPE_BORDER)
        return loupe

    # ----- draw points on preview -----
    def _draw_preview(self):
        preview_copy = self.preview.copy()
        for i, pt in enumerate(self.field_pts):
            x_prev, y_prev = int(pt[0] * self.PREVIEW_SCALE), int(pt[1] * self.PREVIEW_SCALE)
            cv2.circle(preview_copy, (x_prev, y_prev), 5, (0, 255, 0), -1)
            cv2.putText(preview_copy, str(i + 1), (x_prev + 10, y_prev + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        if self.ball_pt1:
            x_prev, y_prev = int(self.ball_pt1[0] * self.PREVIEW_SCALE), int(self.ball_pt1[1] * self.PREVIEW_SCALE)
            cv2.circle(preview_copy, (x_prev, y_prev), 5, (255, 0, 0), -1)
            cv2.putText(preview_copy, "B1", (x_prev + 10, y_prev + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        return preview_copy

    # ----- main loop -----
    def run(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self._mouse)
        cv2.namedWindow("Loupe", cv2.WINDOW_NORMAL)  # Create Loupe window once here

        while True:
            preview_with_points = self._draw_preview()
            cv2.imshow("Image", preview_with_points)

            loupe_image = self._loupe_image()
            cv2.imshow("Loupe", loupe_image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('z') and self.field_pts:
                print(f"Undoing last corner: {self.field_pts.pop()}")
                if not self.field_pts and self.ball_pt1:
                    print(f"Undoing ball 1 point: {self.ball_pt1}")
                    self.ball_pt1 = None

            if len(self.field_pts) == 4 and self.ball_pt1:
                break

        cv2.destroyAllWindows()

        # ---------- homography in cm ----------
        src = np.float32(self.field_pts)
        dst = np.float32([[0, 0],
                            [0, self.FIELD_H_CM],
                            [self.FIELD_W_CM, 0],
                            [self.FIELD_W_CM, self.FIELD_H_CM]])
        H = cv2.getPerspectiveTransform(src, dst)
        ball1 = np.float32([self.ball_pt1]).reshape(-1,1,2)
        x1_cm, y1_cm = cv2.perspectiveTransform(ball1, H)[0,0]
        print(f"Ball 1 → ({x1_cm:.1f} cm, {y1_cm:.1f} cm)")

        # Coordinates of the second ball in cm
        x2_cm, y2_cm = 300.0, 750.0

        # Calculate the distance between the two balls
        delta_x = x2_cm - x1_cm
        delta_y = y2_cm - y1_cm
        distance_cm = math.sqrt(delta_x**2 + delta_y**2)
        print(f"Distance between Ball 1 and Ball 2: {distance_cm:.1f} cm")

        # Calculate the angle relative to the vertical axis
        # The vertical axis corresponds to the y-axis in our real-world coordinates.
        # We want the angle of the vector from ball1 to ball2 with respect to the positive y-axis.
        angle_radians = math.atan2(delta_x, delta_y)  # atan2(x, y) gives angle with respect to positive y-axis
        angle_degrees = math.degrees(angle_radians)

        # Adjust the angle to be within 0 to 360 degrees (optional, for consistency)
        if angle_degrees < 0:
            angle_degrees += 360

        print(f"Angle of distance relative to vertical axis: {angle_degrees:.1f} degrees")

        # optional view (1 px = 1 cm)
        warp_w, warp_h = int(self.FIELD_W_CM), int(self.FIELD_H_CM)
        warped = cv2.warpPerspective(self.img, H, (warp_w, warp_h))
        cv2.circle(warped, (int(x1_cm), int(y1_cm)), 6, (0,0,255), -1)
        cv2.circle(warped, (int(x2_cm), int(y2_cm)), 6, (0,255,0), -1) # Draw second ball in green

        # Draw an arrow representing the distance and angle (optional visualization)
        start_point = (int(x1_cm), int(y1_cm))
        end_point = (int(x2_cm), int(y2_cm))
        cv2.arrowedLine(warped, start_point, end_point, (255, 255, 255), 2)

        cv2.imshow("Warped (1 px = 1 cm)", warped)
        cv2.waitKey(0)

# --------------- run ---------------
if __name__ == "__main__":
    FieldMeasure("WIN_20250508_17_52_53_Pro.jpg").run()