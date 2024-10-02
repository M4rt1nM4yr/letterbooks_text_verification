import os.path
from tqdm import tqdm
from PIL import Image
import re
import csv
import shutil
import time
import multiprocessing

from src.data.utils.fixed_line_height import FixedHeightResize
from src.data.utils.noTransform import NoTransform
from src.data.preprocessing.line_extraction.page_xml_line_extraction import process_img_page_bbox, \
    _point_str_to_tuple_int, get_coords, clamp_values
from src.data.preprocessing.text_processing.clean_nbb_string import clean_string_basic
from src.data.utils.count_chars import count_characters
from src.data.utils.xml_crawler import crawlerXML
from src.utils.constants import *

def preload_meta_nbb_github(root, split):
    raise NotImplementedError()

def preload_all_nbb_github(
        root,
        alphabet,
        books,
        split=None,
        slant_correction=False,
        height=64,
        max_samples=-1,
        color_mode="L",
        return_text = "basic",
        abbreviations=False,
        **kwargs
):
    if return_text == "basic":
        sample_names, meta_dict, image_dict = _load_data(
            root,
            books,
            alphabet,
            height=height,
            max_samples=max_samples,
            color_mode=color_mode,
        )
        out_meta_dict = dict()
        for key, value in meta_dict.items():
            out_meta_dict[key] = {
                "text_dipl": "",
                "text_basic": value["text"],
                "coords": value["coords"],
                "writer": value["writer"],
                "paragraph": value["paragraph"],
            }
            if alphabet is not None:
                out_meta_dict[key]["text_logits_basic"] = alphabet.string_to_logits(out_meta_dict[key]["text_basic"])
                out_meta_dict[key]["text_logits_dipl"] = alphabet.string_to_logits(out_meta_dict[key]["text_dipl"])
        return sample_names, out_meta_dict, image_dict
    elif return_text == "diplomatic":
        return _load_data_dipl(
            root,
            books,
            alphabet,
            height=height,
            max_samples=max_samples,
            color_mode=color_mode,
            abbreviations=abbreviations,
        )
    elif return_text == "both":
        print("Loading diplomatic data")
        sample_names_dipl, meta_dict_dipl, image_dict_dipl = _load_data_dipl(
            root,
            books,
            alphabet,
            height=height,
            max_samples=-1,
            color_mode=color_mode,
            abbreviations=abbreviations,
        )
        print("Loading basic data")
        sample_names_basic, meta_dict_basic, image_dict_basic = _load_data(
            root,
            books,
            alphabet,
            height=height,
            max_samples=-1,
            color_mode=color_mode,
        )
        print(f"Basic samples {len(sample_names_basic)}, Dipl samples {len(sample_names_dipl)}")
        sample_names = list()
        meta_dict = dict()
        image_dict = dict()
        non_match_count = 0
        for key, value in meta_dict_dipl.items():
            if key not in meta_dict_basic.keys():
                print(f"{key} not in meta_dict_basic")
                non_match_count+=1
                continue
            if "?" in meta_dict_basic[key]["text"]:
                print(f"{key} has ? in meta_dict_basic")
                non_match_count+=1
                continue
            sample_names.append(key)
            meta_dict[key] = {
                "text_dipl": value["text"],
                "text_basic": meta_dict_basic[key]["text"],
                "coords": meta_dict_basic[key]["coords"],
                "writer": value["writer"],
                "paragraph": meta_dict_basic[key]["paragraph"],
            }
            if alphabet is not None:
                meta_dict[key]["text_logits_basic"] = alphabet.string_to_logits(meta_dict_basic[key]["text"])
                meta_dict[key]["text_logits_dipl"] = alphabet.string_to_logits(value["text"])
            image_dict[key] = image_dict_basic[key]
        print("non_match_count", non_match_count)

        if max_samples > 0:
            return sample_names[:max_samples], meta_dict, image_dict
        else:
            return sample_names, meta_dict, image_dict
    else:
        raise ValueError(f"return_text {return_text} not supported")

def _load_data(root, books, alphabet, height=64, max_samples=-1, color_mode="RGB"):
    image_dict = dict()
    meta_dict = dict()
    sample_names = list()
    resize = FixedHeightResize(height) if isinstance(height, int) else NoTransform(return_type="pil")
    for book in books:
        print("loading book", book, "...")
        book_root = os.path.join(root, book)
        page_xml_root = os.path.join(root, "nuremberg_letterbooks", "basic", book)
        page_xml_files = os.listdir(page_xml_root)
        page_xml_files.sort()
        page_xml_files = [os.path.join(page_xml_root, f) for f in page_xml_files if os.path.splitext(f)[1]==".xml"]
        possible_img_files = {os.path.splitext(f)[0]: os.path.join(root, book, f) for f in
                              os.listdir(book_root)}
        with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
            results = pool.starmap(
                _load_basic_data_helper,
                tqdm(
                    [(p, possible_img_files, color_mode, resize, alphabet) for p in page_xml_files],
                    total=len(page_xml_files)
                )
            )
        for sample_names_, meta_dict_, image_dict_ in results:
            sample_names.extend(sample_names_)
            meta_dict.update(meta_dict_)
            image_dict.update(image_dict_)

        print(len(sample_names))
    return sample_names, meta_dict, image_dict

def _load_basic_data_helper(
        p,
        possible_img_files,
        color_mode,
        resize,
        alphabet=None,
):
    meta_dict = dict()
    image_dict = dict()
    sample_names = list()
    name = os.path.splitext(os.path.basename(p))[0]
    page_out = process_img_page_bbox(
        img=Image.open(possible_img_files[name]),
        page_xml_path=p,
        color_mode=color_mode,
    )
    for i, (img, meta) in enumerate(page_out):
        text = clean_string_basic(meta["text"].strip())
        if alphabet is not None:
            if not alphabet.check_valid_input_chars(text):
                print(
                    f"{name} | {alphabet.return_wrong_input(text)} | {text}")
                continue
        line_name = meta["id"] if isinstance(meta["id"], str) else name
        image_dict[f"{name}_{line_name}"] = resize(img)
        meta_dict[f"{name}_{line_name}"] = {
            "coords": meta["coords"],
            "text": text,
            "writer": -1,
            "paragraph": name,
        }
        if alphabet is not None:
            meta_dict[f"{name}_{line_name}"]["text_logits"] = alphabet.string_to_logits(text)
        sample_names.append(f"{name}_{line_name}")
    return sample_names, meta_dict, image_dict


def _load_data_dipl(root, books, alphabet, height=64, max_samples=-1, color_mode="RGB", abbreviations=True):
    image_dict = dict()
    meta_dict = dict()
    sample_names = list()
    resize = FixedHeightResize(height) if isinstance(height, int) else NoTransform(return_type="pil")
    sample_limit_reached = False
    for book in books:
        print("loading book", book, "...")
        dipl_norm_xml_root = os.path.join(root, "nuremberg_letterbooks", "diplomatic-regularised", book)
        meta_dipl = _extract_meta_dipl_only(dipl_norm_xml_root)
        possible_img_files = {os.path.splitext(f)[0]:os.path.join(root, book, f) for f in os.listdir(os.path.join(root, book))}

        current_page_image_name = ""
        for key, value in tqdm(meta_dipl.items()):
            if sample_limit_reached:
                break
            if value["page_name"] != current_page_image_name:
                current_page_image_name = value["page_name"]
                page_img = Image.open(possible_img_files[value["page_name"]]).convert(color_mode)
            text = value["text"] if abbreviations else remove_abbreviations(value["text"])
            if alphabet is not None:
                if not alphabet.check_valid_input_chars(text):
                    print(f"{value['page_name']} | {value['letter']} | {alphabet.return_wrong_input(text)} | {text}")
                    continue
            value["text"] = text
            image_dict[key] = resize(page_img.crop(value["coords"]))
            sample_names.append(key)
            meta_dict[key] = value.copy()

    return sample_names, meta_dict, image_dict

def _load_dipl_data_helper(
        key,
        value,
        color_mode,
        possible_img_files,
        resize,
        alphabet=None,
):
    meta_dict = dict()
    image_dict = dict()
    sample_names = list()
    page_img = Image.open(possible_img_files[value["page_name"]]).convert(color_mode)
    text = value["text"]
    if alphabet is not None:
        if not alphabet.check_valid_input_chars(text):
            print(f"{value['page_name']} | {value['letter']} | {alphabet.return_wrong_input(text)} | {text}")
            return sample_names, meta_dict, image_dict
    image_dict[key] = resize(page_img.crop(value["coords"]))
    sample_names.append(key)
    meta_dict[key] = value.copy()
    return sample_names, meta_dict, image_dict



def _extract_meta_dipl_only(root):
    meta_dict = dict()
    files = [os.path.join(root,f) for f in os.listdir(root) if os.path.splitext(f)[1]==".xml"]
    files.sort()
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        results = pool.map(
            _extract_meta_dipl_only_helper,
            tqdm(
                files,
                total=len(files)
            )
        )
    for result in results:
        meta_dict.update(result)
    return meta_dict

def _extract_meta_dipl_only_helper(
        f,
):
    meta_dict = dict()
    name = os.path.splitext(os.path.basename(f))[0]
    page_xml = crawlerXML(f, encoding="utf-8")
    pages = page_xml.find_all("Page")
    for page in pages:
        page_name = os.path.splitext(page["imageFilename"])[0]
        lines = page.find_all("TextLine")
        for line in lines:
            if line.find("TextEquiv") is None:
                continue

            points = _compute_bounding_box(
                line.find("Coords")["points"],
                max_width=int(page["imageWidth"]) - 1,
                max_height=int(page["imageHeight"]) - 1,
            )

            text = str(line.find("TextEquiv").find("Unicode"))
            text_clean = _text_clean_diplomatic(text)

            if "<" in text_clean:
                print("Next errors found", f, page_name, line['id'], text_clean)

            if '?' in text_clean:
                print("Next errors found", f, page_name, line['id'], text_clean)
                continue

            meta_dict[f"{page_name}_{line['id']}"] = {
                "text": text_clean,
                "writer": int(line["writerID"]) if len(line["writerID"].split(",")) == 1 else -1,
                "coords": get_coords(points),
                "page_name": page_name,
                "letter": name,
            }
    return meta_dict

def _compute_bounding_box(coords, max_width, max_height):
    if isinstance(coords, str):
        coords = coords.split(" ")
    points = []
    for point in coords:
        p = _point_str_to_tuple_int(point)
        p = clamp_values(
            p,
            max_x=max_width,
            max_y=max_height,
        )
        points.append([p[0], p[1]])
    return points

def remove_abbreviations(text):
    text = text.replace(EXPAN_OPEN, "")
    text = text.replace(EXPAN_CLOSE, "")
    text = re.sub(f"{EX_OPEN}.*?{EX_CLOSE}", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _reverse_text_clean_diplomatic(text):
    text = text.replace(EXPAN_OPEN, "<expan>")
    text = text.replace(EXPAN_CLOSE, "</expan>")
    text = text.replace(EX_OPEN, "<ex>")
    text = text.replace(EX_CLOSE, "</ex>")
    text = text.replace(HI_OPEN, '<hi rend="superscript">')
    text = text.replace(HI_UNDERLINED_OPEN, '<hi rend="underlined">')
    text = text.replace(HI_CLOSE, "</hi>")
    text = text.replace(DEL_OPEN, '<del rend="strikethrough">')
    text = text.replace(DEL_ERASED_OPEN, '<del rend="erased">')
    text = text.replace(DEL_CLOSE, "</del>")
    text = text.replace(DEL_CROSSOUT_OPEN, '<del rend="crossout">')
    text = text.replace("%", '<gap/>')
    text = text.replace(ADD_OPEN, "<add>")
    text = text.replace(ADD_CLOSE, "</add>")
    return text

def _text_clean_diplomatic(text):
    text = text.replace("\n", " ")
    text = text.replace("<Unicode>","").replace("</Unicode>","").strip()
    text = text.replace("<expan>", EXPAN_OPEN)
    text = text.replace("</expan>", EXPAN_CLOSE)
    text = text.replace("<ex>", EX_OPEN)
    text = text.replace("</ex>", EX_CLOSE)
    text = text.replace('<hi rend="sup">', HI_OPEN)
    text = text.replace('<hi rend="underlined">', HI_UNDERLINED_OPEN)
    text = text.replace('<hi rend="superscript">', HI_OPEN)
    text = text.replace('</hi>', HI_CLOSE)
    text = text.replace('<del rend="strikethrough">', DEL_OPEN)
    text = text.replace('<del rend="erased">', DEL_ERASED_OPEN)
    text = text.replace('<del>', DEL_OPEN)
    text = text.replace('</del>', DEL_CLOSE)
    text = text.replace('<del rend="crossout">', DEL_CROSSOUT_OPEN)
    text = text.replace('<gap/>', "%")
    text = text.replace('<add>', ADD_OPEN)
    text = text.replace('</add>', ADD_CLOSE)
    text = text.replace('<note place="inline">', "") 
    text = text.replace('<note place="bottomcentral">', "") 
    text = text.replace('<note place="interlinear">', "") 
    text = text.replace('<note place="marginleft">', "") 
    text = text.replace('<note place="topright">', "") 
    text = text.replace('</note>', "")
    
    ### remove
    text = text.replace('<expan cert="unknown">', EXPAN_OPEN) 
    text = text.replace('<expan cert="medium">', EXPAN_OPEN) 
    text = text.replace('<expan cert="high">', EXPAN_OPEN)
    text = text.replace('<hi rend="underlined:true;">', HI_UNDERLINED_OPEN)  
    text = text.replace('<gap reason="illegible"/>', "%") 
    text = text.replace('<gap reason="damaged"/>', "%") 
    text = text.replace('<abbr>',EX_OPEN)
    text = text.replace('</abbr>',EX_CLOSE)
    text = text.replace('<expan rend="u">', EXPAN_OPEN)
    text = text.replace('<hi rend="sup>',"")
    text = text.replace('<del/>','')
    text = text.replace('<del rend="strıkethrough>', DEL_OPEN)
    text = text.replace('<del rend="strıkethrough">', DEL_OPEN)
    text = text.replace('<hi rend="sup"/>',"")  
    ### remove
    
    text = re.sub(r'<ab.*?</ab>', '', text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text
