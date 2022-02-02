from RSAD import db
from RSAD.com.vo.LoginVO import LoginVO


class LoginDAO:

    def insertLogin(self, loginVO):
        db.session.add(loginVO)
        db.session.commit()

    def validateLogin(self, loginVO):

        loginList = LoginVO.query.filter_by(
            loginUsername=loginVO.loginUsername, loginPassword=loginVO.loginPassword, loginStatus=loginVO.loginStatus)
        return loginList

    def blockLogin(self, loginVO):
        block = LoginVO.query.filter_by(loginId=loginVO.loginId).first()
        print(block)
        block.loginStatus = 'inactive'
        db.session.commit()

    def unblockLogin(self, loginVO):
        block = LoginVO.query.filter_by(loginId=loginVO.loginId).first()
        block.loginStatus = 'active'
        db.session.commit()

    def deletelogin(self, loginVO):

        loginList = LoginVO.query.get(loginVO.loginId)
        
        db.session.delete(loginList)
        db.session.commit()
        # print(loginVO.login_registrationId)
        # loginList = LoginVO.query.filter_by(loginId=loginVO.loginId).first()
        # print(loginList)
        # db.session.delete(loginList)
        # db.session.commit()

    def updatelogin(self, loginVO):
        db.session.merge(loginVO)

        db.session.commit()
