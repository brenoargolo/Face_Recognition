import face_recognition as fr
import cv2

imgElon = fr.load_image_file('Elon.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTeste = fr.load_image_file('ElonTest.jpg')
imgElonTeste = cv2.cvtColor(imgElonTeste, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0,255,0), 2)

encodingElon = fr.face_encodings(imgElon)[0]
encodingElonTeste = fr.face_encodings(imgElonTeste)[0]

comparacao = fr.compare_faces([encodingElon], encodingElonTeste)
print(comparacao)

cv2.imshow('Elon', imgElon)
cv2.waitKey(0)
