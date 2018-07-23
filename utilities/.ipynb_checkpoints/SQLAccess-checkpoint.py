import MySQLdb as my
import base64
class SQLAccess():
    def __init__(self, host = '127.0.0.1', port = 3306):
        '''
        Writes to our MySQL database. Currently there is no error checking implemented.
        '''
        self.connect(host, port)
    def connect(self, host, port):
        passwd = constants.SQL_PWD
        db = constants.SQL_DB_NAME
        user = constants.SQL_USR
        self.db = my.connect(host, user,passwd, db)
        print('Connected to DB')
    def disconnect(self):
        self.db.close()
        print('Database close')
    def insert(self,lpr_text,car_make_text, car_model_text, car_screenshot_img, car_lpr_screenshot_img, gps_coords):
        cursor = self.db.cursor()
        car_screenshot_img = base64.b64encode(car_screenshot_img)
        car_lpr_screenshot_img = base64.b64encode(car_lpr_screenshot_img)

        sql = "insert into anpr_car_details VALUES('" + lpr_text + "', '" + car_model_text + "', '" + car_model_text + "',%s,%s,'" +datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"','" + gps_coords + "')"

        number_of_rows = cursor.execute(sql,[car_screenshot_img, car_lpr_screenshot_img])
        self.db.commit()   # you need to call commit() method to save 
                      # your changes to the database 
        cursor.close()
        print('Written to DB')