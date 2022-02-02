from RSAD import db

class LoginVO(db.Model):
    __tablename__ = 'loginmaster'
    loginId = db.Column('loginId', db.Integer, primary_key=True, autoincrement=True)
    loginUsername = db.Column('loginUsername', db.String(100), unique=True)
    loginPassword = db.Column('loginPassword', db.String(100))
    loginRole = db.Column('loginRole', db.String(100))
    loginStatus = db.Column('loginStatus', db.String(100))



    def as_dict(self):
        return {
            'loginId': self.loginId,
            'loginUsername': self.loginUsername,
            'loginPassword': self.loginPassword,
            'loginRole':self.loginRole,
            'loginStatus':self.loginStatus,
        }


db.create_all()
