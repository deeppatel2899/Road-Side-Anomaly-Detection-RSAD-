from RSAD import db
from RSAD.com.vo.LoginVO import LoginVO
import datetime


class RegistrationVO(db.Model):
    __tablename__ = 'registrationmaster'
    registrationId = db.Column('registrationId', db.Integer, primary_key=True, autoincrement=True)
    policestationName = db.Column('policestationName', db.String(100))
    policestationCode = db.Column('policestationCode', db.String(100))
    policestationAddress = db.Column('policestationAddress', db.String(100))
    datetime = db.Column('datetime', db.DateTime,default=datetime.datetime.now())
    registration_loginId = db.Column('registration_loginId', db.Integer, db.ForeignKey(LoginVO.loginId))

    def as_dict(self):
        return {
            'registrationId': self.registrationId,
            'policestationName': self.policestationName,
            'policestationCode': self.policestationCode,
            'policestationAddress': self.policestationAddress,
            'datetime': self.datetime,
            'registration_loginId': self.registration_loginId


        }


db.create_all()
