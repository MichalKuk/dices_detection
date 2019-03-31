import numpy as np
import cv2

#liczenie oczek
def calculate_value(roi):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 350

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(roi)

    return len(key_points)


def algorithm(img,imgCopy,finalImage,arrayOfImgs):#img - jeden z obrazów do przetworzenia;  imgCopy - oryginalny, kolorowy obraz;
    # finalImage - na nim nakładane napisy; arrayOfImgs - tablica obrazów - będą na nich zamalowywane policzone kości

    # znajdowanie konturów
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)# RETR_EXTERNAL - tylko kontury-rodzice (zewnętrzne)

    liczba_oczek = 0 #na cały obrazek
    for cnt in contours:
        #znajdowanie najmniejszych prostokątów
        rect = cv2.minAreaRect(cnt)  # ( center (x,y), (width, height), angle of rotation )
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        area = rect[1][0] * rect[1][1]
        if area > 3500 and area < 10000:#warunek dotyczący rozmiaru prostokątów
            aspect_ratio = float(rect[1][0]) / rect[1][1]
            # print("aspect ratio: ", aspect_ratio)
            if aspect_ratio < 0.70 or aspect_ratio > 1.30:# bierzemy tylko ramki zbliżone do kwadratów
                continue

            # stosunek pola prostokąta do convexHull konturu
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            # print("solidity: ", solidity)
            if solidity < 0.90 or solidity > 1.30:
                continue

            # współrzędne nieobróconych ramek do dodawania napisów
            x, y, w, h = cv2.boundingRect(cnt)

            imgToFillPoly = imgCopy.copy()  # nowa kopia oryginału dla każdej iteracji

            # koloruje na czarno obszar kości - środek prostokąta
            filledPoly = cv2.fillConvexPoly(imgToFillPoly, box, (255,255,255))  # zmienia też imgToFillPoly
            filledPolyGray = cv2.cvtColor(filledPoly, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('filledPoly', filledPoly)
            # cv2.imwrite('filledPolyGray.png', filledPolyGray)

        #KOLORUJE NA CZARNO OBSZAR Z POLICZONA JUŻ KOŚCIĄ NA WSZYSTKICH 5 ZDJĘCIACH DO PRZETWORZENIA
            for tmp in arrayOfImgs:
                tmp = cv2.fillConvexPoly(tmp, box, (0,0,0))
                # cv2.imwrite('tmp.png', tmp)
                # cv2.imshow("tmp",tmp)
                # k = cv2.waitKey(0)

            # wszystko co nie jest białe (255) kolorujemy na czarno; zostaje biały obszar kości na czarnym tle
            ret, thresholdedFilledPoly = cv2.threshold(filledPolyGray, 254, 255, cv2.THRESH_BINARY)
            # cv2.imwrite('thresholdedFilledPoly.png', thresholdedFilledPoly)
            # cv2.imshow("thresholdedFilledPoly", thresholdedFilledPoly)

            # nakładany na obraz powyższą maskę, żeby mieć kość z oryginalnego obrazka na czarnym tle
            cutCube = cv2.bitwise_and(imgCopy, imgCopy, mask=thresholdedFilledPoly)
            # cv2.imshow('cutCube', cutCube)
            # cv2.imwrite('cutCube.png', cutCube)

            # liczy bloby na obrazku z jedną kością na czarnym tle
            number = calculate_value(cutCube)
            if number > 0:
                liczba_oczek = liczba_oczek + number
                finalImage = cv2.putText(finalImage, "Oczek: " + str(number), (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 0), 2)
                # cv2.imshow('policzone oczka', finalImage)

            # k = cv2.waitKey(0)

    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return(liczba_oczek)



def main():
    # Wczytywanie obrazu
    imgColor = cv2.imread('cb1.png')
    imgCopy = imgColor.copy()
    finalImage = imgColor.copy()
    # imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

    # rozdzielamy obraz na 3 kanały
    b, g, r = cv2.split(imgColor)
    cv2.imshow("b", b)
    cv2.imshow("g", g)
    cv2.imshow("r", r)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('b.png', b)
    # cv2.imwrite('g.png', g)
    # cv2.imwrite('r.png', r)

    # wyostrzanie
    # kernel_sharpening = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])
    # sharpenedColor = cv2.filter2D(imgColor, -1, kernel_sharpening)
    # cv2.imshow('Image original', imgColor)
    # cv2.imshow('Image Sharpened', sharpenedColor)
    # # cv2.imwrite('sharpenedColor.png', sharpenedColor)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    bluredColour = cv2.GaussianBlur(imgColor, (5, 5), 0)
    bluredB = cv2.GaussianBlur(b, (5, 5), 0)
    bluredG = cv2.GaussianBlur(g, (5, 5), 0)
    bluredR = cv2.GaussianBlur(r, (5, 5), 0)
    cv2.imshow("bluredB", bluredB)
    cv2.imshow("bluredG", bluredG)
    cv2.imshow("bluredR", bluredR)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # thresholding
    thresholdedB = cv2.adaptiveThreshold(bluredB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 6)  # 13,9
    thresholdedG = cv2.adaptiveThreshold(bluredG, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 6)
    thresholdedR = cv2.adaptiveThreshold(bluredR, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 6)
    # thresholdedSharpened = cv2.adaptiveThreshold(cv2.cvtColor(sharpenedColor, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 13)
    cv2.imshow("thresholdedB", thresholdedB)
    cv2.imshow("thresholdedG", thresholdedG)
    cv2.imshow("thresholdedR", thresholdedR)
    # cv2.imshow("thresholdedSharpened", thresholdedSharpened)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('thresholdedB.png', thresholdedB)
    # cv2.imwrite('thresholdedG.png', thresholdedG)
    # cv2.imwrite('thresholdedR.png', thresholdedR)
    # cv2.imwrite('thresholdedSharpened.png', thresholdedSharpened)

    cannyEdgesB = cv2.Canny(thresholdedB, 75, 200)
    cannyEdgesG = cv2.Canny(thresholdedG, 75, 200)
    cannyEdgesR = cv2.Canny(thresholdedR, 75, 200)
    cannyEdgesColor = cv2.Canny(bluredColour, 75, 200)
    # cannyEdgesSharpened = cv2.Canny(thresholdedSharpened, 75, 200)
    cv2.imshow("cannyEdgesB", cannyEdgesB)
    cv2.imshow("cannyEdgesG", cannyEdgesG)
    cv2.imshow("cannyEdgesR", cannyEdgesR)
    # cv2.imshow("cannyEdgesColor", cannyEdgesColor)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('cannyEdgesB.png', cannyEdgesB)
    # cv2.imwrite('cannyEdgesG.png', cannyEdgesG)
    # cv2.imwrite('cannyEdgesR.png', cannyEdgesR)
    # cv2.imwrite('cannyEdgesColor.png', cannyEdgesColor)
    # cv2.imwrite('cannyEdgesSharpened.png', cannyEdgesSharpened)
    # k = cv2.waitKey(0)

    #algorytm na: cannyEdgesB, cannyEdgesG, cannyEdgesR, cannyEdgesColor, thresholdedSharpened
    arrayOfImagesToProcess = [cannyEdgesB, cannyEdgesG, cannyEdgesR, cannyEdgesColor]

    for index,imageToProcess in enumerate(arrayOfImagesToProcess):
        algorithm(imageToProcess, imgCopy, finalImage, arrayOfImagesToProcess)
        print("iteracja: ",index)

    # for indeks, im in enumerate(arrayOfImagesToProcess):
    #     cv2.imshow(str(indeks),im)
    #     k = cv2.waitKey(0)

    cv2.imshow("WYNIK", finalImage)
    # cv2.imwrite('wynik7.png', finalImage)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()