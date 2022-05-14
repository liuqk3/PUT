import cv2
import numpy as np
import random
import os
import random
import matplotlib.pyplot as plt 
from PIL import Image


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.

    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.

    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.

    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """

    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False

    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success


def generate_mask_based_on_landmark(im_size, landmark_coord, landmark_area, maxVertex=3, maxBrushWidth=24, area_margin=0.5):
    """
    im_size: tuple, (h, w)
    landmark_coord: list of tuple (x, y)
    landmark_area: [x1, y1, x2, y2]
    """
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)

    if np.random.rand() > 0.5:
        # generate mask based on landmark coordinates
        count = np.random.randint(1, len(landmark_coord)+1)
        mask += np_coord_form_mask(random.sample(landmark_coord, k=count), maxBrushWidth=maxBrushWidth, h=im_size[0], w=im_size[1])

    else:
        # generate mask based on landmark area
        width = landmark_area[2] - landmark_area[0]
        height = landmark_area[3] - landmark_area[1]
        x_margin = width * area_margin
        y_margin = height * area_margin
        area = [
            int(max(0, landmark_area[0]-x_margin)),
            int(max(0, landmark_area[1]-y_margin)),
            int(min(im_size[1]-1, landmark_area[2]+x_margin)),
            int(min(im_size[0]-1, landmark_area[3]+y_margin))
            ]
        mask_area = generate_stroke_mask((area[3]-area[1], area[2]-area[0]), maxVertex=maxVertex, maxBrushWidth=maxBrushWidth)
        mask[area[1]:area[3], area[0]:area[2], :] = mask_area
    mask = np.minimum(mask, 1.0)

    return mask

def generate_stroke_mask(
        im_size, 
        max_parts=5, 
        maxVertex=25, 
        maxLength=100, 
        maxBrushWidth=24, 
        maxAngle=360, 
        min_parts=1, 
        minVertex=1, 
        minBrushWidth=10, 
        keep_ratio=None,
        min_area=64,
        keep_topk=-1,
        maxRectangle=0, # the max number of rectangles to be masked
        minRectangle=0, # the min number of rectangles to be masked
        maxRectangleRatio=0.8, # the max ratio of the rectangle
        minRectangleRatio=0.1, # the min ratio of the rectangle
    ):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    
    if keep_ratio is None:
        parts = random.randint(min_parts, max_parts)
        # print('number parts: ', parts)
        for i in range(parts):
            mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1], 
                                            minVertex=minVertex,
                                            minBrushWidth=minBrushWidth,
                                            maxRectangle=maxRectangle,
                                            minRectangle=minRectangle,
                                            maxRectangleRatio=maxRectangleRatio,
                                            minRectangleRatio=minRectangleRatio)
    else:
        assert isinstance(keep_ratio, (list, tuple)) and keep_ratio[0] >= 0 and keep_ratio[1] <= 1.0 and keep_ratio[0] < keep_ratio[1]
        keep_ratio_ = random.uniform(keep_ratio[0], keep_ratio[1])
        while np.sum((mask > 0).astype(np.float32)) / (im_size[0]*im_size[1]) < keep_ratio_:
            mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1], 
                                            minVertex=minVertex,
                                            minBrushWidth=minBrushWidth,
                                            maxRectangle=maxRectangle,
                                            minRectangle=minRectangle,
                                            maxRectangleRatio=maxRectangleRatio,
                                            minRectangleRatio=minRectangleRatio)
    mask = np.minimum(mask, 1.0)
    # mask = np.concatenate([mask, mask, mask], axis = 2)

    # import pdb; pdb.set_trace()
    # remove some small holes
    mask = fill_small_holes(mask, fill_value=1, area_threshold=min_area)
    mask = fill_small_holes(mask, fill_value=1, area_threshold=min_area, keep_topk=keep_topk)
    return mask

def np_free_form_mask(
        maxVertex, 
        maxLength, 
        maxBrushWidth, 
        maxAngle, 
        h, w, 
        minVertex=1, 
        minBrushWidth=10,
        maxRectangle=0, # the max number of rectangles to be masked
        minRectangle=0, # the min number of rectangles to be masked
        maxRectangleRatio=0.7, # the max ratio of the rectangle
        minRectangleRatio=0.1, # the min ratio of the rectangle
    ):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(minVertex, maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)

    # draw some lines with value 1
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        # print('length: ', length)
        brushWidth = np.random.randint(minBrushWidth, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)       
        cv2.line(mask, (startX, startY), (nextX, nextY), 1, brushWidth)
        cv2.circle(mask, (startX, startY), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    #cv2.circle(mask, (startY, startX), brushWidth // 2, 2)

    # draw some rectangles with value 0
    num_rectangles = random.randint(minRectangle, maxRectangle)
    for i in range(num_rectangles):
        # get rectangle w and h
        rw = random.randint(int(w*minRectangleRatio), int(w*maxRectangleRatio))
        rh = random.randint(int(h*minRectangleRatio), int(h*maxRectangleRatio))

        # get the position of rectangle
        x1 = random.randint(0, w-rw)
        y1 = random.randint(0, h-rh)
        mask[y1:y1+rh, x1:x1+rw,:] = 0
    mask = np.minimum(mask, 1.0)
    return mask

def np_coord_form_mask(coords, maxBrushWidth, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    for c in coords:
        c = tuple(c)
        brushWidth = np.random.randint(12, maxBrushWidth + 1) // 2 * 2
        mask_type = np.random.choice(['circle', 'rectangle', 'ellipse'])
        # mask_type = 'ellipse'
        if mask_type == 'circle':
            cv2.circle(mask, c, brushWidth // 2, 2, -1)
        elif mask_type == 'ellipse':
            long_axis = int((1 + np.random.rand() * 0.5) * brushWidth) // 2
            short_axis = int((1 - np.random.rand() * 0.5) * brushWidth) // 2
            rotate_angle = np.random.randint(0, 180)
            cv2.ellipse(mask, c, (long_axis, short_axis), rotate_angle, 0, 360, 2, thickness=-1)
        else:
            max_ = int((1 + np.random.rand() * 0.5) * brushWidth)
            min_ = int((1 - np.random.rand() * 0.5) * brushWidth)
            h_ = np.random.randint(min_, max_)
            w_ = np.random.randint(min_, max_)
            pt1 = (max(0, c[0]-w_//2), max(0, c[1]-h_//2))
            pt2 = (min(w-1, c[0]+w_//2), min(h-1, c[1]+h_//2))
            cv2.rectangle(mask, pt1, pt2, 2, thickness=-1)
    mask = np.minimum(mask, 1.0)
    return mask



def fill_small_holes(mask, fill_value, area_threshold=64, keep_topk=-1, show_contour=False, show_result=False):
    """
        mask: np.array, 2D or 3D
    """
    if len(mask.shape) == 2:
        mask_find = mask.copy().astype(np.uint8)
        mask_return = mask.copy().astype(np.uint8)
    elif len(mask.shape) == 3 and mask.shape[-1] == 1:
        mask_find = mask[:, :, 0].copy().astype(np.uint8)
        mask_return = mask[:, :, 0].copy().astype(np.uint8)
    elif len(mask.shape) == 3 and mask.shape[-1] == 3:
        mask_find = np.sum(mask, axis=-1).astype(np.uint8)
        mask_return = mask.copy().astype(np.uint8)
        if isinstance(fill_value, (int, float)):
            fill_value = (fill_value, fill_value, fill_value)
    else:
        raise ValueError('Not supported data type')

    contours, hierarchy = cv2.findContours(mask_find.copy().astype(np.bool).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if show_contour:
        #show the contours of the imput image
        if len(mask.shape) == 2:
            mask_show = mask.copy().astype(np.uint8)
        elif len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask_show = mask[:,:,0].copy().astype(np.uint8)
        else:
            mask_show = mask.copy().astype(np.uint8)

        for i in range(len(contours)):
            color = (random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
            cv2.drawContours(mask_show, contours[i:i+1], -1, color, 2)
        plt.figure('original image with contours')
        plt.imshow(mask_show, cmap = 'gray')

    # print('keep top k: {}, num contours {}'.format(keep_topk, len(contours)))
    
    # keep_topk = 0
    if keep_topk > 0 and len(contours) > 1:
        keep_topk = min(keep_topk, len(contours)-1)
        contours_area = [cv2.contourArea(c) for c in contours]
        idx = np.argsort(contours_area).tolist()
        keep_idx = idx[-keep_topk:]
    else:
        keep_idx = list(range(len(contours)))

    # import pdb; pdb.set_trace()
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i])
        if a < area_threshold or i not in keep_idx:
            cv2.fillConvexPoly(mask_return, contours[i], fill_value)

    if show_result:
        plt.figure('result image after filling small holes')
        plt.imshow(mask_return)
    
    if show_contour or show_result:
        plt.show()

    mask_return = mask_return.astype(mask.dtype)
    if len(mask.shape) == 3 and mask.shape[-1] == 1:
        mask_return = mask_return[:, :, np.newaxis].astype(mask.dtype)
    
    return mask_return


def rgba_to_depth(x):
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[2] == 4
    y = x.copy()
    y.dtype = np.float32
    y = y.reshape(x.shape[:2])
    return np.ascontiguousarray(y)


def visualize_depth(depth):
    if isinstance(depth, str):
        rgba = np.array(Image.open(depth))
        depth = rgba_to_depth(rgba)
    depth = (depth - depth.min())/max(1e-8, depth.max()-depth.min()) # in range [0, 1]
    depth = depth * 255.0
    return depth



# def dilate_demo(image, k=8):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
#     dst = cv2.dilate(binary, kernel=kernel)
#     return dst

# def erode_demo(image, k=8):
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
#     dst = cv2.erode(binary, kernel=kernel)
#     return dst



if __name__ == '__main__':

    mask = cv2.imread('mask.png')
    # if len(mask.shape) == 3:
    #     mask = mask[:, :, 0]
    fill_small_holes(mask, fill_value=255, keep_topk=1, area_threshold=64, show_contour=True, show_result=True)






    # import cv2  
    # img = cv2.imread("mask.png")  
    # img = dilate_demo(img)
    # # cv2.imshow("img", img)  
    # im = Image.fromarray(img)
    # im.show()

    # img = erode_demo(img)
    # # cv2.imshow("img", img)  
    # im = Image.fromarray(img)
    # im.show()