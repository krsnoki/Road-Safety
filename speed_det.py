import cv2
import dlib
import time
import math

speed_limit = 50


def speed_show(image, speed, x1, y1, w1, h1):
    """Shows speed of vehicle"""
    cv2.putText(
        image,
        f"{speed} km/h",
        (int(x1 + w1 / 2), int(y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 100, 0),
        2,
    )


def warn_show(image, speed, x1, y1, w1, h1):
    """Warns of overspeeding"""
    cv2.putText(
        image,
        f"WARNING",
        (int(x1 + w1 / 2), int(y1 - 35)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 190),
        3,
    )
    speed_show(image, speed, x1, y1, w1, h1)


def estimateSpeed(location1, location2):
    """Estimates speed on pixel distance"""
    d_pixels = math.sqrt(
        math.pow(location2[0] - location1[0], 2)
        + math.pow(location2[1] - location1[1], 2)
    )
    ppm = 16
    d_meters = d_pixels / ppm
    fps = 10
    speed = (d_pixels * fps) * 3.6
    # print(d_pixels)
    return speed


def main():
    # constants used in code
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    carCascade = cv2.CascadeClassifier("cascades/vech.xml")
    human_cascade = cv2.CascadeClassifier("cascades/haarcascade_fullbody.xml")

    video_file_name = "videos/vehicle_example_1.mp4"
    video = cv2.VideoCapture(video_file_name)

    first = True
    while True:
        rc, image = video.read()
        if type(image) == type(None):
            break
        resultImage = image.copy()
        humans = human_cascade.detectMultiScale(resultImage, 1.9, 2, None, (15, 15))
        # warns when humans are found in frame
        if len(humans):
            cv2.putText(
                resultImage,
                "HUMAN FOUND",
                (23, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                2,
            )
        elif len(humans) == 0:
            cv2.putText(
                resultImage,
                "No Human",
                (23, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 0),
                2,
            )

        # Display the resulting frame with humans
        for (x, y, w, h) in humans:
            cv2.rectangle(
                resultImage,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )

        frameCounter = frameCounter + 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)
        # remove cars with low quality tracking
        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        # run only on 10th frame
        if frameCounter % 10 == 0:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # find cars in frame
            cars = carCascade.detectMultiScale(grey, 1.1, 13, 18, (20, 20))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if (
                        (t_x <= x_bar <= (t_x + t_w))
                        and (t_y <= y_bar <= (t_y + t_h))
                        and (x <= t_x_bar <= (x + w))
                        and (y <= t_y_bar <= (y + h))
                    ):
                        matchCarID = carID

                if matchCarID is None:
                    # track the found cars
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(
                resultImage,
                (t_x, t_y),
                (t_x + t_w, t_y + t_h),
                rectangleColor,
                4,
            )

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        # last location and current location of car is now
        # known to find pixle distance and then find speed
        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if speed[i] == None or speed[i] == 0:
                        speed[i] = estimateSpeed([x1, y1], [x1, y2])
                        continue

                    if speed[i] >= speed_limit:
                        warn_show(resultImage, speed[i], x1, y1, w1, h1)
                        print(f"WARNING overspeeding: {speed[i]}")
                    elif speed[i] != None:
                        speed_show(resultImage, speed[i], x1, y1, w1, h1)

        cv2.imshow("result", resultImage)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
