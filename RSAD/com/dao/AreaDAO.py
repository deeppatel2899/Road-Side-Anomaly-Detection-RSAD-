from RSAD import db
from RSAD.com.vo.AreaVO import AreaVO


class AreaDAO:

    def insertArea(self, areaVo):
        db.session.add(areaVo)
        db.session.commit()

    def viewArea(self):
        areaList=AreaVO.query.all()

        return areaList

    def deleteArea(self,areaVO):

        areaList = AreaVO.query.get(areaVO.areaId)

        db.session.delete(areaList)

        db.session.commit()

    def editArea(self,areaVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        areaList = AreaVO.query.filter_by(areaId=areaVO.areaId).all()

        return areaList

    def updateArea(self,areaVO):

        db.session.merge(areaVO)

        db.session.commit()
