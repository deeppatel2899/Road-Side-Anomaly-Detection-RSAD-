from RSAD import db
from RSAD.com.vo.RegistrationVO import RegistrationVO
from RSAD.com.vo.LoginVO import LoginVO


class RegistrationDAO:

    def insertRegistration(self, RegistrationVo):
        db.session.add(RegistrationVo)
        db.session.commit()

    def viewRegistration(self):
        #        RegistrationList=RegistrationVO.query.all()
        registrationList = db.session.query(LoginVO, RegistrationVO).join(
            LoginVO, RegistrationVO.registration_loginId == LoginVO.loginId).all()
        return registrationList

    def deleteRegistration(self, registrationVO):

        # registrationList = RegistrationVO.query.get(registrationVO.registration_loginId)
        # print(registrationList)
        # db.session.delete(registrationList)

        # db.session.commit()
        registrationList = RegistrationVO.query.filter_by(registration_loginId=registrationVO.registration_loginId).first()
        print(registrationList)
        db.session.delete(registrationList)
        db.session.commit()

    def editRegistration(self, registrationVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        # registrationList = RegistrationVO.query.filter_by(registrationId=registrationVO.registrationId).all()
        registrationList = db.session.query(LoginVO, RegistrationVO).join(
            LoginVO, RegistrationVO.registration_loginId == LoginVO.loginId).filter_by(loginId=registrationVO.registration_loginId).all()

        return registrationList

    def updateRegistration(self, registrationVO):

        db.session.merge(registrationVO)

        db.session.commit()
