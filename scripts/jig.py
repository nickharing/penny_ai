import cv2
import numpy as np
import pickle

class CoinProcessor:
    def __init__(self, image_path, max_display_size=920):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not open image at {image_path}")
        
        # Image properties
        self.max_display_size = max_display_size
        self.display_image, self.display_scale = self.resize_for_display(self.original_image)
        
        # Processing state
        self.warped_image = None
        self.selected_coin = None
        self.circle_radius = 200
        self.circle_center = (100, 100)
        self.rotation_angle = 0
        self.rois = {'date': None, 'mint_mark': None, 'liberty': None}
        self.normalized_rois = {'date': None, 'mint_mark': None, 'liberty': None}
        
        # Constants
        self.RADIUS_CHANGE = 2
        self.POSITION_CHANGE = 5
        self.ROTATION_CHANGE = 1   # 1° steps
        
        print(f"--- Penny Measurement Jig ---")
        print(f"Image: {self.image_path} ({self.original_image.shape[1]}x{self.original_image.shape[0]})")
        print(f"Display Size: {self.display_image.shape[1]}x{self.display_image.shape[0]} (Scale: {self.display_scale:.3f})")
        print(f"Output File: {self.image_path.split('.')[0]}_data.pkl")
        print(f"------------------------------")
    
    def resize_for_display(self, image):
        h, w = image.shape[:2]
        scale = 1.0
        max_dim = max(h, w)
        if max_dim > self.max_display_size:
            scale = self.max_display_size / max_dim
            return cv2.resize(image, (int(w*scale), int(h*scale))), scale
        return image, scale
    
    def display_to_original(self, x, y):
        return int(x / self.display_scale), int(y / self.display_scale)
    
    def original_to_display(self, x, y):
        return int(x * self.display_scale), int(y * self.display_scale)
    
    def correct_warp(self):
        corners, needs_redraw = [], True
        win = "Select Four Corners (Clockwise from TL)"
        def on_mouse(evt, x, y, flags, param):
            nonlocal needs_redraw
            if evt == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                ox, oy = self.display_to_original(x, y)
                corners.append((ox, oy))
                print(f"Perspective point {len(corners)} added (Orig: {ox},{oy})")
                needs_redraw = True
        
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while len(corners) < 4:
            if needs_redraw:
                disp = self.display_image.copy()
                for i, (ox, oy) in enumerate(corners):
                    dx, dy = self.original_to_display(ox, oy)
                    cv2.circle(disp, (dx, dy), 4, (0,255,0),1)
                    cv2.putText(disp, str(i+1), (dx+5, dy+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
                for i, txt in enumerate([
                    "Click four corners",
                    "Press U to undo",
                    "Press ESC to cancel"
                ]):
                    cv2.putText(disp, txt, (10, 30+i*20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                cv2.imshow(win, disp)
                needs_redraw = False
            
            key = cv2.waitKeyEx(1)
            if key == 27:
                cv2.destroyWindow(win)
                return False
            if key in (ord('u'), ord('U')) and corners:
                rem = corners.pop()
                print(f"Removed point {len(corners)+1}: {rem}")
                needs_redraw = True
        
        cv2.destroyWindow(win)
        src = np.array(corners, dtype="float32")
        h = int(round(max(np.linalg.norm(src[0]-src[3]),
                          np.linalg.norm(src[1]-src[2]))))
        w = int(round(max(np.linalg.norm(src[0]-src[1]),
                          np.linalg.norm(src[2]-src[3]))))
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        self.warped_image = cv2.warpPerspective(self.original_image, M, (w, h))
        print(f"Warped to {w}x{h}")
        disp_w, _ = self.resize_for_display(self.warped_image)
        cv2.imshow("Perspective Corrected", disp_w)
        cv2.waitKey(1000)
        return True
    
    def move_circle(self, dx, dy):
        x = int(np.clip(self.circle_center[0]+dx, 0, self.warped_image.shape[1]))
        y = int(np.clip(self.circle_center[1]+dy, 0, self.warped_image.shape[0]))
        self.circle_center = (x, y)
        print(f"Moved circle to: {self.circle_center}")
    
    def resize_circle(self, dr):
        self.circle_radius = max(5, self.circle_radius+dr)
        print(f"Resized circle to radius: {self.circle_radius}")
    
    def rotate_circle(self, da):
        self.rotation_angle = (self.rotation_angle - da) % 360
        print(f"Rotated circle to: {self.rotation_angle}°")
    
    def extract_upper_left_coin(self):
        if self.warped_image is None:
            print("Please correct warp first"); return False
        
        disp_w, scale = self.resize_for_display(self.warped_image)
        win, needs_redraw = "Adjust Coin", True
        
        def draw_dashed_circle(img, center, radius, color, segments=60):
            step = 360 / segments
            for i in range(segments):
                start_ang = np.deg2rad(i*step)
                end_ang   = np.deg2rad(i*step + step/2)
                p1 = (int(center[0] + radius*np.cos(start_ang)),
                      int(center[1] + radius*np.sin(start_ang)))
                p2 = (int(center[0] + radius*np.cos(end_ang)),
                      int(center[1] + radius*np.sin(end_ang)))
                cv2.line(img, p1, p2, color, 1)
        
        key_actions = {
            2490368: lambda: self.move_circle(0, -self.POSITION_CHANGE),   # Up arrow
            2621440: lambda: self.move_circle(0, self.POSITION_CHANGE),    # Down arrow
            2424832: lambda: self.move_circle(-self.POSITION_CHANGE, 0),   # Left arrow
            2555904: lambda: self.move_circle(self.POSITION_CHANGE, 0),    # Right arrow
            ord('+'):   lambda: self.resize_circle(self.RADIUS_CHANGE),
            ord('='):   lambda: self.resize_circle(self.RADIUS_CHANGE),
            ord('-'):   lambda: self.resize_circle(-self.RADIUS_CHANGE),
            ord('1'):   lambda: self.rotate_circle(self.ROTATION_CHANGE),
            ord('2'):   lambda: self.rotate_circle(-self.ROTATION_CHANGE),
        }
        
        def on_mouse(evt, x, y, flags, param):
            nonlocal needs_redraw
            if evt == cv2.EVENT_LBUTTONDOWN:
                ox, oy = int(x/scale), int(y/scale)
                self.circle_center = (ox, oy)
                print(f"Set center to: {self.circle_center}")
                needs_redraw = True
        
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        
        while True:
            if needs_redraw:
                img = disp_w.copy()
                cc = (int(self.circle_center[0]*scale),
                      int(self.circle_center[1]*scale))
                cr = int(self.circle_radius * scale)
                
                draw_dashed_circle(img, cc, cr, (0,255,255))
                ang = np.deg2rad(self.rotation_angle - 90)  # 12 o'clock base
                end_pt = (int(cc[0] + cr*np.cos(ang)),
                          int(cc[1] + cr*np.sin(ang)))
                cv2.line(img, cc, end_pt, (0,255,255), 1)
                
                y0 = 30
                for txt in ["Arrows: Move", "+/-: Resize", "1/2: Rotate", "Enter: Confirm", "Esc: Cancel"]:
                    cv2.putText(img, txt, (10,y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                    y0 += 20
                status = f"C: {self.circle_center} R: {self.circle_radius} A: {self.rotation_angle}°"
                cv2.putText(img, status, (10, img.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                
                cv2.imshow(win, img)
                needs_redraw = False
            
            key = cv2.waitKeyEx(1)
            if key == 13:  # Enter
                break
            if key == 27:
                cv2.destroyWindow(win)
                return False
            if key in key_actions:
                needs_redraw = True
                key_actions[key]()
            elif key != -1:
                print(f"Key {key}")
        
        cv2.destroyWindow(win)
        
        # finalize extraction
        H, W = self.warped_image.shape[:2]
        cx, cy = self.circle_center
        M = cv2.getRotationMatrix2D((cx, cy), self.rotation_angle, 1.0)
        rotated = cv2.warpAffine(self.warped_image, M, (W, H),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255,255,255))
        
        x1, y1 = cx-self.circle_radius, cy-self.circle_radius
        x2, y2 = x1+2*self.circle_radius, y1+2*self.circle_radius
        x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
        region = rotated[y1:y2, x1:x2].copy()
        
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (self.circle_radius, self.circle_radius),
                   self.circle_radius, 255, -1)
        bg = np.full_like(region, 255)
        bg[mask==255] = region[mask==255]
        self.selected_coin = bg
        
        disp_coin, _ = self.resize_for_display(self.selected_coin)
        cv2.imshow("Extracted Coin", disp_coin)
        cv2.waitKey(1000)
        return True

    def define_rois(self):
        if self.selected_coin is None:
            print("Please extract a coin first")
            return False
        
        disp_coin, scale = self.resize_for_display(self.selected_coin)
        diameter = 2 * self.circle_radius
        
        for name in ['date', 'mint_mark', 'liberty']:
            win = f"Select {name.upper()}"
            start = end = None
            drawing = False
            needs_redraw = True
            
            def on_mouse(evt, x, y, flags, param):
                nonlocal start, end, drawing, needs_redraw
                if evt == cv2.EVENT_LBUTTONDOWN:
                    start = (x, y)
                    end = (x, y)
                    drawing = True
                    needs_redraw = True
                elif evt == cv2.EVENT_MOUSEMOVE and drawing:
                    end = (x, y)
                    needs_redraw = True
                elif evt == cv2.EVENT_LBUTTONUP and drawing:
                    end = (x, y)
                    drawing = False
                    needs_redraw = True
            
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, on_mouse)
            
            while True:
                if needs_redraw:
                    img = disp_coin.copy()
                    if start and end:
                        x1, y1 = min(start[0], end[0]), min(start[1], end[1])
                        x2, y2 = max(start[0], end[0]), max(start[1], end[1])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 1)
                    y0 = 30
                    for txt in ["Drag to select", "Enter: OK", "Esc: Skip"]:
                        cv2.putText(img, txt, (10, y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        y0 += 20
                    cv2.imshow(win, img)
                    needs_redraw = False
                
                key = cv2.waitKeyEx(1)
                if key == 27:  # skip
                    self.rois[name] = (0,0,0,0)
                    self.normalized_rois[name] = (0,0,0,0)
                    break
                if key == 13 and start and end:
                    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
                    w, h = abs(end[0] - start[0]), abs(end[1] - start[1])
                    ox, oy = int(x1/scale), int(y1/scale)
                    ow, oh = int(w/scale), int(h/scale)
                    self.rois[name] = (ox, oy, ow, oh)
                    self.normalized_rois[name] = (ox/diameter, oy/diameter, ow/diameter, oh/diameter)
                    print(f"{name} ROI: {self.rois[name]}")
                    break
            
            cv2.destroyWindow(win)
            if self.rois[name][2:] != (0, 0):
                x, y, w, h = self.rois[name]
                roi = self.selected_coin[y:y+h, x:x+w]
                if roi.size:
                    dr, _ = self.resize_for_display(roi)
                    cv2.imshow(name.upper(), dr)
                    cv2.waitKey(500)
                    cv2.destroyWindow(name.upper())
        
        return True
    
    def save_data(self, output_file):
        data = {
            'diameter': 2 * self.circle_radius,
            'offset': self.rotation_angle,
            'roi_coordinates': self.rois,
            'normalized_roi_coordinates': self.normalized_rois,
            'center': self.circle_center
        }
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {output_file}")
        return True

def main():
    try:
        path = input("Image path (Enter for default): ")
        if not path:
            path = r"C:\Users\nickh\OneDrive\Documents\nh\Git_repo\coin_sorter3\image_20250429_092630.png"
        ms = input("Max display size (default 920): ")
        max_sz = int(ms) if ms.isdigit() else 920
        out = path.split('.')[0] + "_data.pkl"
        
        proc = CoinProcessor(path, max_display_size=max_sz)
        if proc.correct_warp() and proc.extract_upper_left_coin() and proc.define_rois():
            proc.save_data(out)
            print("Done.")
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
