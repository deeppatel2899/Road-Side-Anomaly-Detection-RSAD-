from RSAD import db
from RSAD.com.vo.FeedbackVO import FeedbackVO

class FeedbackDAO:

    def insertFeedback(self, feedbackVo):
        db.session.add(feedbackVo)
        db.session.commit()

    def viewFeedback(self):

        feedbackList=FeedbackVO.query.all()
        return feedbackList

    def viewFeedbackFrom(self, feedbackVO):

        # categoryList = CategoryVO.query.get(categoryVO.categoryId)

        # categoryList = CategoryVO.query.filter_by(categoryId=categoryVO.categoryId)

        feedbackList = FeedbackVO.query.filter_by(
            feedbackFromLogin=feedbackVO.feedbackFromLogin).all()

        return feedbackList


    def deleteFeedback(self,feedbackVO):

        feedbackList = FeedbackVO.query.get(feedbackVO.feedbackId)
        db.session.delete(feedbackList)
        db.session.commit()

    def updateFeedback(self, feedbackVO):

        db.session.merge(feedbackVO)

        db.session.commit()
