import cv2
import numpy as np
from screeninfo import get_monitors
import os


def uploud(i):
    i = i - ord('0')
    image = norm_size(cv2.imread('pliki/{}'.format(images[i])))
    cv2.imshow('obrazek', image)
    return image


def resize(img, s):
    h, w = img.shape[:2]
    h = h + int(h * s)
    w = w + int(w * s)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def norm_size(img):
    screen = get_monitors()[0]
    h, w = img.shape[:2]
    if h > screen.height:
        s = (1 - (screen.height / h)) * (-1)
        img = resize(img, s)
    h, w = img.shape[:2]
    if w > screen.width:
        s = (1 - (screen.width / w)) * (-1)
        img = resize(img, s)
    return img


def hsv_range():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, low_color])
    upper = np.array([255, 255, high_color])

    mask = cv2.inRange(hsv_frame, lower, upper)
    cv2.imshow('obrazek', mask)


def hsv_bitwais():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('obrazek', res)


def hsv_median():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.medianBlur(res, ksize=ksize)
    cv2.imshow('obrazek', res)


def morphology():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((4, 4), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('obrazek', mask_without_noise)


def morphology2():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])
    mask = cv2.inRange(hsv_frame, lower, upper)
    kernel = np.ones((4, 4), np.uint8)
    mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_without_noise, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('obrazek', mask_closed)


def marker():
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([low_color, 100, 100])
    upper = np.array([high_color, 255, 255])

    mask = cv2.inRange(hsv_frame, lower, upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('obrazek', image_marker)


def ball_save():
    blured = cv2.blur(image, (50, 50))
    hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

    lower_color = np.array([0, 75, 75])
    upper_color = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    image_marker = image.copy()
    cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
        0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.circle(image_marker, (cx, cy), 5, (255, 0, 0), -1)
    cv2.imshow('obrazek', image_marker)


def movie_save():
    video = cv2.VideoCapture()
    video.open('movingball.mp4')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(
        'result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    counter = 1

    while True:
        success, frame_rgb = video.read()
        if not success:
            break
        print('Frame {} from {}'.format(counter, total_frames))

        blured = cv2.blur(frame_rgb, (50, 50))
        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

        lower_color = np.array([0, 75, 75])
        upper_color = np.array([15, 255, 255])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        image_marker = frame_rgb.copy()
        cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
            0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.circle(image_marker, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imshow('obrazek', image_marker)
        result.write(image_marker)
        counter = counter + 1

    video.release()
    result.release()


def crow_save():
    video = cv2.VideoCapture()
    video.open('Crow.mp4')
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(
        'crow_result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

    counter = 1

    while True:
        success, frame_rgb = video.read()
        if not success:
            break
        print('Frame {} from {}'.format(counter, total_frames))

        blured = cv2.blur(frame_rgb, (10, 10))
        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([255, 255, 50])

        mask = cv2.inRange(hsv, lower_color, upper_color)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        image_marker = frame_rgb.copy()
        cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
            0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.circle(image_marker, (cx, cy), 5, (255, 0, 0), -1)
        cv2.imshow('obrazek', image_marker)
        result.write(image_marker)
        counter = counter + 1

    video.release()
    result.release()


def tray_detect():
    img1 = image[:, :, 0]
    img2 = image[:, :, 2]

    c_img = image.copy()

    diff = cv2.absdiff(img1, img2)

    thresh = cv2.threshold(diff, 125, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = cv2.contourArea(contours[0])
    print(area)
    end = cv2.drawContours(c_img, [contours[0]], -1, (255, 0, 0), thickness=2)
    cv2.imshow('filled_mask', end)


def coin_detect():
    img1 = image[:, :, 0]
    img2 = image[:, :, 2]

    diff = cv2.absdiff(img1, img2)
    thresh = cv2.threshold(diff, 125, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

    c_img = image.copy()
    w_img = image.copy()

    w_img[opening != 0] = [0, 0, 0]

    gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30, param1=15, param2=30, minRadius=10, maxRadius=50)

    total_area = 0
    fiver = 0
    nickel = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        max_radius = np.max(circles[:, 2])
        min_radius = np.min(circles[:, 2])
        for (x, y, r) in circles:
            total_area += np.pi * r ** 2
            if r > ((max_radius+min_radius)/2):
                cv2.circle(c_img, (x, y), r, (0, 0, 255), 2)
                fiver += 1
            else:
                cv2.circle(c_img, (x, y), r, (0, 255, 0), 2)
                nickel += 1

    cv2.imshow("coins", c_img)
    print("Total area of circles:", total_area)
    print("Total number of fivers:", fiver)
    print("Total number of nickels:", nickel)

def coin_tray():

    img1 = image[:, :, 0]
    img2 = image[:, :, 2]

    diff = cv2.absdiff(img1, img2)

    thresh = cv2.threshold(diff, 125, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    tray_area = cv2.contourArea(contours[0])

    cc_image = image.copy()
    mask = np.zeros_like(cc_image)
    tray_mask = cv2.drawContours(mask, [contours[0]], -1, (255, 255, 255), thickness=-1)
    c_img = cv2.drawContours(cc_image, [contours[0]], -1, (255, 0, 0), thickness=2)
    w_img = image.copy()

    w_img[opening != 0] = [0, 0, 0]

    gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30, param1=15, param2=30, minRadius=10, maxRadius=50)

    total_area = 0
    fiver = 0
    nickel = 0
    tray_fiver = 0
    tray_nickel = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        max_radius = np.max(circles[:, 2])
        min_radius = np.min(circles[:, 2])
        fiver_area = np.pi * max_radius ** 2
        for (x, y, r) in circles:
            total_area += np.pi * r ** 2
            if r > ((max_radius+min_radius)/2):
                cv2.circle(c_img, (x, y), r, (0, 0, 255), 2)
                if tray_mask[y,x].any():
                    tray_fiver += 1
                else:
                    fiver += 1
            else:
                cv2.circle(c_img, (x, y), r, (0, 255, 0), 2)
                if tray_mask[y,x].any():
                    tray_nickel += 1
                else:
                    nickel += 1

    cv2.imshow("coins", c_img)
    print("Total area of circles:", total_area)
    print("Total number of fivers:", fiver+tray_fiver)
    print("Total number of nickels:", nickel+tray_nickel)
    print("Tray area:", tray_area)
    print("Tray to Fiver ratio", tray_area/fiver_area)
    print("Tray money", tray_fiver*5 + tray_nickel*0.05)
    print("Table Money", fiver*5 + nickel*0.05)
    print("")


def temp():
    img1 = image[:, :, 0]
    img2 = image[:, :, 2]

    diff = cv2.absdiff(img1, img2)

    thresh = cv2.threshold(diff, 125, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening, 50, 150)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    area = cv2.contourArea(contours[0])
    print(area)

    mask = np.zeros_like(image)
    tray_mask= cv2.drawContours(mask, [contours[0]], -1, (255, 255, 255), thickness=-1)

    cv2.imshow('filled_mask', tray_mask)


def change_h(x):
    global fun
    if fun is not None:
        fun()


images = ['ball.png']
image = None
fun = None


def main():
    global image, images, fun

    images = os.listdir('pliki/')
    image = norm_size(cv2.imread('pliki/{}'.format(images[0])))
    nimg = image.copy()
    cv2.imshow('obrazek', image)
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
        # -----------image----------------
        if key >= ord('0') and key <= ord('9'):
            image = uploud(key)
            nimg = image.copy()
        # ----------------resolution---------------
        elif key == ord('-'):
            image = resize(image, -0.1)
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('+'):
            image = resize(image, 0.1)
            nimg = image.copy()
            cv2.imshow('obrazek', image)
        elif key == ord('='):
            cv2.imshow('obrazek', image)
            nimg = image.copy()
        # ----------------colors------------------------
        elif key == ord('q'):
            cv2.imshow('obrazek', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        elif key == ord('w'):
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('obrazek', nimg)
        elif key == ord('e'):
            hsv_range()
            fun = hsv_range
        elif key == ord('r'):
            hsv_bitwais()
            fun = hsv_bitwais
        elif key == ord('t'):
            hsv_median()
            fun = hsv_median
        elif key == ord('z'):
            # h = barwa
            cv2.imshow('obrazek', nimg[:, :, 0])
        elif key == ord('x'):
            # s = nasycene
            cv2.imshow('obrazek', nimg[:, :, 1])
        elif key == ord('c'):
            # v = wartość
            cv2.imshow('obrazek', nimg[:, :, 2])
        # ----------------filters------------------------
        elif key == ord('a'):
            cv2.imshow('obrazek', cv2.Canny(image, 55.0, 30.0))
        elif key == ord('s'):
            cv2.imshow('obrazek', cv2.blur(image, (7, 7)))
        elif key == ord('d'):
            b = cv2.blur(image, (7, 7))
            cv2.imshow('obrazek', cv2.Canny(b, 55.0, 30.0))
        elif key == ord('f'):
            morphology()
            fun = morphology
        elif key == ord('g'):
            morphology2()
            fun = morphology
        elif key == ord('h'):
            marker()
            fun = marker
            # ----------------functions------------------------
        elif key == ord('p'):
            ball_save()
            fun = ball_save
        elif key == ord('o'):
            movie_save()
            fun = movie_save
        elif key == ord('i'):
            crow_save()
            fun = crow_save
        elif key == ord("m"):
            tray_detect()
            fun = tray_detect
        elif key == ord("n"):
            coin_detect()
            fun = coin_detect
        elif key == ord("b"):
            coin_tray()
            fun = coin_tray
        elif key == ord("v"):
            temp()
            fun = temp
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
