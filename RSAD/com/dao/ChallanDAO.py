from RSAD import db
from RSAD.com.vo.ChallanVO import ChallanVO
from RSAD.com.vo.uploadchallanVO import VideoVO



class ChallanDAO:

    def insertChallan(self, challanVo):
        db.session.add(challanVo)
        db.session.commit()

    def viewChallan(self):
#        ChallanList = db.session.query(ChallanVO, VideoVO).join(VideoVO,ChallanVO.Challan_VideoId == VideoVO.categoryId).all()

        challanList=db.session.query(ChallanVO,VideoVO).join(VideoVO,ChallanVO.Challan_videoId == VideoVO.videoId).all()

        return challanList

    def deleteChallan(self,challanVO):

        challanList = ChallanVO.query.get(challanVO.challanId)

        db.session.delete(challanList)

        db.session.commit()

    def editChallan(self,challanVO):

        # categoryList = VideoVO.query.get(categoryVO.categoryId)
        # categoryList = VideoVO.query.filter_by(categoryId=categoryVO.categoryId)

        challanList = ChallanVO.query.filter_by(challanId=challanVO.challanId).all()

        return challanList

    def updateChallan(self,challanVO):

        db.session.merge(challanVO)
        db.session.commit()


    def ajaxChallanProduct(self, challanVO):
        ajaxProductchallanList = challanVO.query.filter_by(Challan_VideoId=ChallanVO.Challan_VideoId).all()
        return ajaxProductchallanList
