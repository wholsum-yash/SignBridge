import json
from collections import OrderedDict

def count_videos_per_gloss(json_path):
    
    # Loading the JSON file having gloss and video_id
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # OrderedDict to keep track of counts while potentially preserving order
    gloss_video_count = OrderedDict()
    
    # looping through each sign entry in the dataset
    for item in data:
        gloss = item['gloss']
        instances = item.get('instances', [])
        
        # Using set to store video_id to ensure we only count unique videos from each gloss
        video_ids = set()
        
        # Collect all video_id for the current sign
        for instance in instances:
            video_id = instance.get('video_id')
            if video_id:
                video_ids.add(video_id)
        
        # Map the gloss name to the count of unique videos found
        gloss_video_count[gloss] = len(video_ids)
    
    return gloss_video_count

if __name__ == '__main__':
    # Define the path to the WLASL dataset file
    json_path = '/home/wholsum/projects/SignBridge/WLASL-complete/WLASL_v0.3.json'
    
    # Perform the counting logic
    gloss_video_count = count_videos_per_gloss(json_path)
    
    # Display overall summary statistics
    print(f"Total glosses: {len(gloss_video_count)}")
    print(f"Total unique videos: {sum(gloss_video_count.values())}")
    
    # Sort and display the results to see which signs have the fewest/most videos
    print("\nVideos per gloss (sorted by count, increasing):")
    sorted_glosses = sorted(gloss_video_count.items(), key=lambda x: x[1])
    
    for gloss, count in sorted_glosses:
        print(f"{gloss}: {count} video(s)")
