from ocr_utils import extrai_texto_e_indexa, _carica_ocr_metadata

txt = extrai_texto_e_indexa(r"C:\Users\JUBIS\Desktop\LETTURA ETIQUECHE\test.jpeg")
print(txt)
print(_carica_ocr_metadata()[r"C:\Users\JUBIS\Desktop\LETTURA ETIQUECHE\test.jpeg"])



import pytesseract, cv2
text = pytesseract.image_to_string(cv2.imread(r"C:\Users\JUBIS\Desktop\LETTURA ETIQUECHE\test.jpeg"))
print(text)
