import numpy as np
from PIL import Image, ImageOps
from kraken.lib.segmentation import extract_polygons
import torch
from torchvision import transforms

from src.data.utils.xml_crawler import crawlerXML


def _point_str_to_tuple_int(point_str, seperator=","):
    """
    :param point_str: str like "1,2"
    :param seperator: str
    :return: tuple of ints like (1,2)
    """
    x, y = point_str.split(seperator)
    return int(x), int(y)


def extract_line_meta(page_xml):
    """
    :param page_xml: BeautifulSoup
    :return: dict like: {"type": "baselines",
                            "lines": [{"baseline":[points], "boundary":[points]}, ...]
                        }
    """
    out = list()
    max_x = int(page_xml.find("Page")["imageWidth"])-1
    max_y = int(page_xml.find("Page")["imageHeight"])-1
    for line in page_xml.find_all("TextLine"):
        baseline = list()
        boundary = list()
        for point in line.find("Baseline")["points"].split(" "):
            p = _point_str_to_tuple_int(point)
            p = clamp_values(p, max_x, max_y)
            baseline.append([p[0], p[1]])
        for point in line.find("Coords")["points"].split(" "):
            p = _point_str_to_tuple_int(point)
            p = clamp_values(p, max_x, max_y)
            boundary.append([p[0], p[1]])
        out.append({"baseline":baseline, "boundary": boundary})
    return {"type": "baselines", "lines": out}


def extract_line_meta_bbox(page_xml):
    """
    :param page_xml: BeautifulSoup
    """
    out = list()
    max_x = int(page_xml.find("Page")["imageWidth"])-1
    max_y = int(page_xml.find("Page")["imageHeight"])-1
    for line in page_xml.find_all("TextLine"):
        if line.find("TextEquiv") is None:
            print(page_xml.find("Page")["imageFilename"] ,line["id"])
            continue

        points = list()
        text = line.find("TextEquiv").find("Unicode").decode_contents()
        for point in line.find("Coords")["points"].split(" "):
            p = _point_str_to_tuple_int(point)
            p = clamp_values(p, max_x, max_y)
            points.append([p[0], p[1]])
        coords = get_coords(points)
        out.append({
            "coords": coords,
            "text": text,
            "font": page_xml.find("Font").string.strip() if page_xml.find("Font") is not None else None,
            "id": line["id"] if "id" in line.attrs else None,
        }) # coords = (left, upper, right, bottom)
    return out

def get_coords(points):
    left, upper = np.min(np.array(points), axis=0)
    right, bottom = np.max(np.array(points), axis=0)
    return (left, upper, right, bottom)

def clamp_values(p, max_x=2322, max_y=3408):
    return max(0, min(p[0], max_x)), max(0, min(p[1], max_y))

def process_img_page(img, page_xml_path):
    """
    :param img: PIL.Image
    :param page_xml_path: str
    """
    img = img.convert("L")
    page_xml = crawlerXML(page_xml_path, encoding="utf-8")
    meta = extract_line_meta(page_xml)
    return extract_lines_with_baseline(img, meta)

def process_img_page_bbox(img, page_xml_path, color_mode="L", max_samples=-1):
    """
    :param img: PIL.Image
    :param page_xml_path: str
    """
    img = img.convert(color_mode)
    page_xml = crawlerXML(page_xml_path, encoding="utf-8")
    meta = extract_line_meta_bbox(page_xml)
    line_imgs = list()
    for line in meta:
        line_imgs.append(img.crop(line["coords"]))
        if max_samples>0:
            if len(line_imgs)>=max_samples:
                break
    return [(im, m) for im, m in zip(line_imgs, meta)]


def extract_lines_with_baseline(im, meta):
    """
    :param im: PIL.Image
    :param meta: {"type": "baselines",
                    "lines": [{
                        'baseline': [[1041,122], [1187,118], [1330,114], [1468,110], [1610,106], [1748,102], [1891,98], [2029,94], [2171,85], [2322,81]],
                        'boundary': [[2320,25], [2113,32], [2026,37], [1889,41], [1747,45], [1686,47], [1039,69], [1041,138], [1188,134], [1330,130], [1468,126], [1611,122], [1749,118], [1891,114],
                                     [2030,110], [2172,102], [2322,98], [2320,25]],
                    }]
                    }
    :return: list of tuples (images, baselines, boundaries)
    """
    im_t = transforms.ToTensor()(im)
    if torch.mean(im_t)>0.5:  # white background
        im = ImageOps.invert(im)
    result = extract_polygons(im, meta)
    out = list()
    for r in result:
        out.append(r)
    return out

