from RSAD import db
import datetime

class ComplainVO(db.Model):
    __tablename__ = 'complainmaster'
    complainId = db.Column('complainId', db.Integer, primary_key=True, autoincrement=True)
    complainSubject = db.Column('complainSubject', db.String(100))
    complainDescription = db.Column('complainDescription', db.String(500))
    complainFilepath = db.Column('complainFilepath', db.String(100))
    complainDate = db.Column('complainDate', db.String(100), default=datetime.datetime.now().strftime("%d-%m-%Y"))
    complainTime = db.Column('complainTime', db.String(100), default=datetime.datetime.now().strftime("%H:%M:%S"))
    complainFromLogin = db.Column('complainFromLogin', db.Integer)
    complainToLogin = db.Column('complainToLogin', db.Integer)
    replyFilepath = db.Column('replyFilepath', db.String(100))
    replySubject = db.Column('replySubject', db.String(100))
    replyDescription = db.Column('replyDescription', db.String(500))
    replyDate = db.Column('replyDate', db.String(100))
    replyTime = db.Column('replyTime', db.String(100))


    def as_dict(self):
        return {
            'complainId': self.complainId,
            'complainSubject': self.complainSubject,
            'complainDescription': self.complainDescription,
            'complainFilepath': self.complainFilepath,
            'complainToLogin': self.complainToLogin,
            'complainFromLogin':self.complainFromLogin,
            'replySubject': self.replySubject,
            'replyDescription': self.replyDescription,
            'replyFilepath': self.replyFilepath,
            'complainDate': self.complainDate,
            'complainTime': self.complainTime,
            'replyDate': self.replyDate,
            'replyTime': self.replyTime

        }


db.create_all()
