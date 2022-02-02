from RSAD import db
from RSAD.com.vo.ComplainVO import ComplainVO


class ComplainDAO:

    def insertComplain(self, complainVo):
        db.session.add(complainVo)
        db.session.commit()

    def viewComplain(self):

        complainList = ComplainVO.query.all()
        return complainList

    def deleteComplain(self, complainVO):

        complainList = ComplainVO.query.get(complainVO.complainId)
        db.session.delete(complainList)
        db.session.commit()

    def replyComplain(self, complainVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        complainList = ComplainVO.query.filter_by(complainId=complainVO.complainId).all()

        return complainList

    def viewComplainFrom(self, complainVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        complainList = ComplainVO.query.filter_by(complainFromLogin=complainVO.complainFromLogin).all()

        return complainList

    def viewReplayFrom(self, complainVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        replayList = ComplainVO.query.filter_by(complainFromLogin=complainVO.complainFromLogin, complainId=complainVO.complainId).all()

        return replayList

    def updateComplain(self,complainVO):

        db.session.merge(complainVO)

        db.session.commit()
