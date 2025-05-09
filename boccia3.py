import cv2
import numpy as np
import math


class FieldMeasure:
    # ---------- settings ----------
    PREVIEW_SCALE = 0.5       # main preview size
    LOUPE_SCALE = 2.0       # magnification inside loupe
    LOUPE_WIN = 200       # loupe window side (px)
    FIELD_W_CM = 400.0     # set real width here
    FIELD_H_CM = 500.0     # set real height here
    LOUPE_BORDER = 2         # Border thickness for loupe
    LOUPE_BORDER_COLOR = (0, 255, 0)  # Green border
    TARGET_COLOR = (0, 0, 255)       # Red color for the target dot
    TARGET_RADIUS = 2
    WHITE_COLOR_LOWER = np.array([190, 200, 200], np.uint8)  # Tunable: Lower bound for white in BGR
    WHITE_COLOR_UPPER = np.array([255, 255, 255], np.uint8)  # Tunable: Upper bound for white in BGR
    MIN_BALL_RADIUS = 5       # Tunable: Minimum radius (in pixels) to consider a ball
    BALL_CIRCULARITY_THRESHOLD = 0.5  # Tunable: Minimum circularity to consider a ball
    # --------------------------------

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(img_path)

        self.preview = cv2.resize(
            self.img, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
            interpolation=cv2.INTER_AREA)

        self.field_pts, self.ball_pt1 = [], None
        self.cursor_preview = (0, 0)  # Cursor coordinates on the preview image
        self.ball_detected = False

    def _get_threshold_mask(self, image):
        return cv2.inRange(image, self.WHITE_COLOR_LOWER, self.WHITE_COLOR_UPPER)

    def _detect_ball(self):
        img_bgr = self.img.copy()
        mask = self._get_threshold_mask(img_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        max_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    ((x, y), radius) = cv2.minEnclosingCircle(contour)

                    if radius > self.MIN_BALL_RADIUS and circularity > self.BALL_CIRCULARITY_THRESHOLD:
                        if circularity > max_circularity:
                            max_circularity = circularity
                            best_ball = (int(x), int(y))

        if best_ball:
            self.ball_pt1 = best_ball
            self.ball_detected = True
            print(f"White ball detected at → ({self.ball_pt1[0]}, {self.ball_pt1[1]}) with circularity: {max_circularity:.2f}")
        else:
            print("No suitable white ball detected based on size and circularity.")

    def _mouse(self, e, x, y, flags, _):
        self.cursor_preview = (x, y)
        if e == cv2.EVENT_LBUTTONDOWN:
            X = int(x / self.PREVIEW_SCALE)
            Y = int(y / self.PREVIEW_SCALE)
            if len(self.field_pts) < 4:
                self.field_pts.append((X, Y))
                print(f"Corner {len(self.field_pts)} → ({X}, {Y})")

    def _loupe_image(self):
        x_p, y_p = self.cursor_preview
        X = int(x_p / self.PREVIEW_SCALE)
        Y = int(y_p / self.PREVIEW_SCALE)

        half = int(self.LOUPE_WIN / (2 * self.LOUPE_SCALE))
        x1, y1 = max(0, X - half), max(0, Y - half)
        x2, y2 = min(self.img.shape[1], X + half), min(self.img.shape[0], Y + half)
        patch = self.img[y1:y2, x1:x2]
        loupe = cv2.resize(patch, (self.LOUPE_WIN, self.LOUPE_WIN), interpolation=cv2.INTER_NEAREST)
        center = self.LOUPE_WIN // 2
        cv2.circle(loupe, (center, center), self.TARGET_RADIUS, self.TARGET_COLOR, -1)
        cv2.rectangle(loupe, (0, 0), (self.LOUPE_WIN - 1, self.LOUPE_WIN - 1), self.LOUPE_BORDER_COLOR, self.LOUPE_BORDER)
        return loupe

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

    def run(self):
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self._mouse)
        cv2.namedWindow("Loupe", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)  # New window for threshold

        while len(self.field_pts) < 4:
            preview_with_points = self._draw_preview()
            cv2.imshow("Image", preview_with_points)
            cv2.imshow("Loupe", self._loupe_image())

            # Show the thresholded image
            threshold_image = self._get_threshold_mask(self.img)
            cv2.imshow("Threshold", threshold_image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('z') and self.field_pts:
                print(f"Undoing last corner: {self.field_pts.pop()}")
            # Adjust WHITE_COLOR_LOWER individually
            elif key == ord('1'):  # Increment lower Blue
                self.WHITE_COLOR_LOWER[0] = np.clip(self.WHITE_COLOR_LOWER[0] + 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            elif key == ord('a'):  # Decrement lower Blue
                self.WHITE_COLOR_LOWER[0] = np.clip(self.WHITE_COLOR_LOWER[0] - 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            elif key == ord('2'):  # Increment lower Green
                self.WHITE_COLOR_LOWER[1] = np.clip(self.WHITE_COLOR_LOWER[1] + 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            elif key == ord('s'):  # Decrement lower Green
                self.WHITE_COLOR_LOWER[1] = np.clip(self.WHITE_COLOR_LOWER[1] - 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            elif key == ord('3'):  # Increment lower Red
                self.WHITE_COLOR_LOWER[2] = np.clip(self.WHITE_COLOR_LOWER[2] + 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            elif key == ord('d'):  # Decrement lower Red
                self.WHITE_COLOR_LOWER[2] = np.clip(self.WHITE_COLOR_LOWER[2] - 5, 0, 255)
                print(f"WHITE_COLOR_LOWER: {self.WHITE_COLOR_LOWER}")
            # Adjust WHITE_COLOR_UPPER individually
            elif key == ord('4'):  # Increment upper Blue
                self.WHITE_COLOR_UPPER[0] = np.clip(self.WHITE_COLOR_UPPER[0] + 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")
            elif key == ord('f'):  # Decrement upper Blue
                self.WHITE_COLOR_UPPER[0] = np.clip(self.WHITE_COLOR_UPPER[0] - 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")
            elif key == ord('5'):  # Increment upper Green
                self.WHITE_COLOR_UPPER[1] = np.clip(self.WHITE_COLOR_UPPER[1] + 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")
            elif key == ord('g'):  # Decrement upper Green
                self.WHITE_COLOR_UPPER[1] = np.clip(self.WHITE_COLOR_UPPER[1] - 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")
            elif key == ord('6'):  # Increment upper Red
                self.WHITE_COLOR_UPPER[2] = np.clip(self.WHITE_COLOR_UPPER[2] + 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")
            elif key == ord('h'):  # Decrement upper Red
                self.WHITE_COLOR_UPPER[2] = np.clip(self.WHITE_COLOR_UPPER[2] - 5, 0, 255)
                print(f"WHITE_COLOR_UPPER: {self.WHITE_COLOR_UPPER}")

        cv2.destroyAllWindows()

        if len(self.field_pts) == 4:
            self._detect_ball()

        if len(self.field_pts) == 4 and self.ball_pt1:
            src = np.float32(self.field_pts)
            dst = np.float32([[0, 0], [0, self.FIELD_H_CM], [self.FIELD_W_CM, 0], [self.FIELD_W_CM, self.FIELD_H_CM]])
            H = cv2.getPerspectiveTransform(src, dst)
            self.ball_pt1 = (self.ball_pt1[0], self.ball_pt1[1] + 5)
            ball1 = np.float32([self.ball_pt1]).reshape(-1, 1, 2)
            x1_cm, y1_cm = cv2.perspectiveTransform(ball1, H)[0, 0]
            print(f"Ball 1 → ({x1_cm:.1f} cm, {y1_cm:.1f} cm)")

            x2_cm, y2_cm = 300.0, 750.0
            delta_x = x2_cm - x1_cm
            delta_y = y2_cm - y1_cm
            distance_cm = math.sqrt(delta_x**2 + delta_y**2)
            print(f"Distance between Ball 1 and Ball 2: {distance_cm:.1f} cm")

            angle_radians = math.atan2(delta_x, delta_y)
            angle_degrees = math.degrees(angle_radians)
            if angle_degrees < 0:
                angle_degrees += 360
            print(f"Angle of distance relative to vertical axis: {angle_degrees:.1f} degrees")

            warp_w, warp_h = int(self.FIELD_W_CM), int(self.FIELD_H_CM)
            warped = cv2.warpPerspective(self.img, H, (warp_w, warp_h))
            cv2.circle(warped, (int(x1_cm), int(y1_cm)), 6, (0, 0, 255), -1)
            cv2.circle(warped, (int(x2_cm), int(y2_cm)), 6, (0, 255, 0), -1)
            cv2.arrowedLine(warped, (int(x1_cm), int(y1_cm)), (int(x2_cm), int(y2_cm)), (255, 255, 255), 2)
            cv2.imshow("Warped (1 px = 1 cm)", warped)
            cv2.waitKey(0)
        else:
            print("Could not proceed with measurement as the field corners or the ball were not detected.")


if __name__ == "__main__":
    FieldMeasure("WIN_20250508_17_52_53_Pro.jpg").run()
