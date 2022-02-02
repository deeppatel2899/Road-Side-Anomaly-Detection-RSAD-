from RSAD import db
from RSAD.com.vo.CrossroadVO import CrossroadVO
from RSAD.com.vo.AreaVO import AreaVO



class CrossroadDAO:

    def insertCrossroad(self, crossroadVo):
        db.session.add(crossroadVo)
        db.session.commit()

    def viewCrossroad(self):
#        CrossroadList = db.session.query(CrossroadVO, AreaVO).join(AreaVO,CrossroadVO.Crossroad_AreaId == AreaVO.categoryId).all()

        crossroadList=db.session.query(CrossroadVO,AreaVO).join(AreaVO,CrossroadVO.crossroad_areaId == AreaVO.areaId).all()

        return crossroadList

    def deleteCrossroad(self,crossroadVO):

        crossroadList = CrossroadVO.query.get(crossroadVO.crossroadId)

        db.session.delete(crossroadList)

        db.session.commit()

    def editCrossroad(self,crossroadVO):

        # categoryList = AreaVO.query.get(categoryVO.categoryId)

        # categoryList = AreaVO.query.filter_by(categoryId=categoryVO.categoryId)

        crossroadList = CrossroadVO.query.filter_by(crossroadId=crossroadVO.crossroadId).all()

        return crossroadList

    def updateCrossroad(self,crossroadVO):

        db.session.merge(crossroadVO)

        db.session.commit()


    def ajaxCrossroadProduct(self, crossroadVO):
        ajaxProductcrossroadList = crossroadVO.query.filter_by(crossroad_areaId=CrossroadVO.crossroad_areaId).all()
        return ajaxProductcrossroadList
