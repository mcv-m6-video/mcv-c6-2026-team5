import xml.etree.ElementTree as ET
from collections import defaultdict

def load_gt_xml(xml_path, exclude_parked=True):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = defaultdict(list)

    for track in root.findall('track'):

        # Only cars&bikes
        if track.attrib.get('label') not in ['car', 'bike']:
            continue

        for box in track.findall('box'):

            # Skip if outside frame
            if box.attrib.get('outside') == '1':
                continue
            
            is_parked = False
            for attr in box.findall('attribute'):
                if attr.attrib.get('name') == 'parked':
                    if attr.text == 'true':
                        is_parked = True
                        break

            if exclude_parked and is_parked:
                continue

            frame_id = int(box.attrib['frame'])

            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            w = xbr - xtl
            h = ybr - ytl

            gt_boxes[frame_id].append([xtl, ytl, w, h])

    return gt_boxes