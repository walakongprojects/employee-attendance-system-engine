import cv2
import os
import pymongo

cam = cv2.VideoCapture(0)

cv2.namedWindow("Capture Pictures")

img_counter = 0

# Collect info from employee
employee_name = input("Enter the full name of the employee : ")
employee_age = input("Enter the age of the employee : ")
employee_number = input("Enter the employee number : ")
os.mkdir("./train_img/"+ employee_name)

#insert employee in database
# dev -----
# myclient = pymongo.MongoClient("mongodb://localhost:27017/")
# mydb = myclient["emp-attendance-dev"]
# prod -----
myclient = pymongo.MongoClient("mongodb+srv://niconiconi:niconiconi@cluster0.qskbw.mongodb.net/attendace-prod?retryWrites=true&w=majority")
mydb = myclient["attendace-prod"]

mycol = mydb["employees"]

mydict = { "name": employee_name, "employeeNumber": employee_number, "age": employee_age }

while True and img_counter <= 10:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "ActiOn_{}.jpg".format(img_counter)
        # cv2.imwrite(img_name, frame)
        cv2.imwrite("train_img/"+employee_name+"/"+img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

x = mycol.insert_one(mydict)
print(x.inserted_id)

print(employee_name + " is inserted on database")

cam.release()

cv2.destroyAllWindows()