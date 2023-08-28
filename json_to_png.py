import json
import os
import shutil
from PIL import Image, ImageDraw

METAINFO = dict(
    classes=('background','mound', 'light', 'build', 'wall', 'steel', 'tank',
             'cement', 'boat', 'tower', 'jvma',
             'Stone', 'tree', 'river', 'car', 'truck', 'bus', 'train',
             'motorcycle', 'bicycle'),
    palette=[[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
             [190, 153, 153], [153, 153, 153], [250, 170,
                                                30], [220, 220, 0],
             [107, 142, 35], [152, 251, 152], [70, 130, 180],
             [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])


def json_to_png(json_file_path, output_folder):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    image = Image.new('RGB', (image_width, image_height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        if len(points)<3:
            continue
        uuple_points = [(point[0], point[1]) for point in points]
        id=0
        mask_color = METAINFO['palette'][0]
        for l in METAINFO['classes']:
            if label == l:
                mask_color = tuple(METAINFO['palette'][id])
            id+=1

        draw.polygon(uuple_points, fill=mask_color)

    output_image_path = os.path.join(output_folder, os.path.splitext(os.path.basename(json_file_path))[0] + '.png')
    image.save(output_image_path)
    print(f'Image saved to {output_image_path}')

input_folder = './data/'
output_folder = './dataset/'

def all_to_png():
    # 遍历文件夹内所有 JSON 文件
    for filename in os.listdir(output_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(output_folder, filename)
            json_to_png(json_file_path, output_folder)



def move_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_file_path = os.path.join(input_folder, filename)
            jpg_file_path = os.path.join(input_folder, os.path.splitext(filename)[0] + '.jpg')

            if os.path.exists(jpg_file_path):
                destination_json = os.path.join(output_folder, filename)
                destination_jpg = os.path.join(output_folder, os.path.basename(jpg_file_path))

                # 移动文件并覆盖同名文件
                shutil.move(json_file_path, destination_json)
                shutil.move(jpg_file_path, destination_jpg)
                print(f'Moved and replaced {filename} and its corresponding JPG in {output_folder}')


#move_files(input_folder, output_folder)
all_to_png()