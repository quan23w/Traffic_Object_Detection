import cv2 
import torch
import numpy as np
from ultralytics.solutions.solutions import BaseSolution, SolutionResults, SolutionAnnotator
import supervision as sv
from shapely.geometry import Point, Polygon
region_color = [(255,0,255),
                (0,255,255),
                (85,0,253),
                (0,128,255),
                (235,183,52),
                (255,34, 134)]
vehicle_color = [(255, 0, 0), 
                (0, 255, 0),
                (0, 0, 128)]
class MultipleObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg_regions = self.CFG["regions"] if "regions" in self.CFG else self.CFG.get("region", None)
        self.regions = cfg_regions
        
        self.in_counts = [0] * len(self.regions)
        self.out_counts = [0] * len(self.regions)
        self.counted_ids = [set() for _ in range(len(self.regions))]
        
        self.region_initialized = False
        self.show_in = self.CFG['show_in']
        self.show_out = self.CFG['show_out']
    
    def initialize_region_geometry(self):
        self.Point = Point
        self.Polygon = Polygon
        
        
    def count_object_in_region(self, region_idx, region_points, current_centroid,track_id, prev_position):
        if prev_position is None or track_id in self.counted_ids[region_idx]:
            return 
        polygon = self.Polygon(region_points)
        if polygon.contains(self.Point(current_centroid)):
            xs = [point[0] for point in region_points]
            ys = [point[1] for point in region_points]
            region_width = max(xs) - min(xs)
            region_height = max(ys) - min(ys)
            
            going_in = False
            if region_width <region_height and current_centroid[0]> prev_position[0]:
                going_in = True
            elif region_width >= region_height and current_centroid[1] > prev_position[1]:
                going_in = True
                
            if going_in:
                self.in_counts[region_idx] += 1
            else:
                self.out_counts[region_idx] += 1
            
            self.counted_ids[region_idx].add(track_id)
    
    def display_count(self, plot_im):
        
        for i ,region_points in enumerate(self.regions):
            xs = [point[0] for point in region_points]
            ys = [point[1] for point in region_points]
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            
            text_str = f"{self.in_counts[i]+self.out_counts[i]}"
            
            cv2.putText(plot_im, 
                        text_str,
                        org=(cx, cy),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5, 
                        color=(255, 255, 255), thickness=4)
        
    def process(self,frame):
        if not  self.region_initialized:
            self.initialize_region_geometry()
            self.region_initialized = True
            
        self.extract_tracks(frame)
        
        self.annonator = SolutionAnnotator(im=frame, line_width=self.line_width)
        
        
        for idx, region_points in enumerate(self.regions):
            color = region_color[idx]
            self.annonator.draw_region(reg_pts = region_points,
                                        color=color, 
                                        thickness=self.line_width*2)
            
            b, g, r = color
            frame = sv.draw_filled_polygon(scene = frame,
                                            polygon = np.array(region_points),
                                            color=sv.Color(r=r, g=g, b=b), 
                                            opacity=0.25)
            
            
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annonator.box_label(
                box=box,
                label=self.names[cls],
                color=vehicle_color[object_classes.index(cls)]
            )
            
            self.store_tracking_history(track_id, box)
            
            current_centroid = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2,
            )
            
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
                
            for region_idx, region_points in enumerate(self.regions):
                self.count_object_in_region(region_idx,
                                            region_points,
                                            current_centroid,
                                            track_id, 
                                            prev_position)
        plot_im = self.annonator.result()
        
        self.display_count(plot_im)
        
        self.display_output(plot_im)
            
        return SolutionResults(
            plot_im = plot_im,
            total_tracks = len(self.track_ids)
        )



if __name__ == "__main__":
    object_classes = [2,5,7] #car, truck , bus
    cap = cv2.VideoCapture("sample/video.mp4")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Video writer
    # Video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    video_writer = cv2.VideoWriter("vehicle_count.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                
    region_points = [   
        
        [[441, 554], [225, 557], [506, 247], [568, 247]],
        [[221, 554], [16, 551], [434, 246], [500, 247]],
        [[15, 548], [7, 495], [7, 435], [374, 242], [428, 244]],
        [[658, 239], [715, 239], [958, 554], [769, 554]],
        [[715, 239], [769, 239], [1138, 554], [960, 554]],
        [[1141, 551], [1275, 554], [1271, 521], [817, 239], [773, 239]]
    ]
    
    
    counter = MultipleObjectCounter(
        show = True,
        region = region_points,
        model = 'model/yolo11n.pt',
        classes = object_classes
    )               
                
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = counter(frame)
        print("Frame shape:", frame.shape)
        frame = results.plot_im
        video_writer.write(frame)                             
                

                    
                    
                    
                    
                
                
  