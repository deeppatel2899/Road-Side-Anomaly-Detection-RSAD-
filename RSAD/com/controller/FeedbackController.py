from flask import request, render_template, redirect, url_for, session
from RSAD import app
from RSAD.com.dao.FeedbackDAO import FeedbackDAO
from RSAD.com.vo.FeedbackVO import FeedbackVO
from RSAD.com.controller.LoginController import LoginSession, LogoutSession
from RSAD.com.vo.LoginVO import LoginVO
import datetime


@app.route('/user/postFeedback', methods=['POST'])
def userInsertFeedback():
    try:
        if LoginSession() == 'user':

            feedbackVO = FeedbackVO()
            feedbackDAO = FeedbackDAO()

            feedbackSubject = request.form['feedbackSubject']
            feedbackDescription = request.form['feedbackDescription']
            feedbackRating = request.form['feedbackRating']
            print(feedbackRating)
            feedbackVO.feedbackFromLogin = session['session_loginId']
            feedbackVO.feedbackSubject = feedbackSubject
            feedbackVO.feedbackDescription = feedbackDescription
            feedbackVO.feedbackRating = feedbackRating
            feedbackDAO.insertFeedback(feedbackVO)
            return redirect(url_for('userViewFeedback'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/user/viewFeedback', methods=['GET'])
def userViewFeedback():
    try:
        if LoginSession() == 'user':
            feedbackDAO = FeedbackDAO()
            feedbackVO = FeedbackVO()
            feedbackVO.feedbackFromLogin = session['session_loginId']
            feedbackVOList = feedbackDAO.viewFeedbackFrom(feedbackVO)

            print("__________________", feedbackVOList)
            return render_template('user/viewFeedback.html', feedbackVOList=feedbackVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/viewFeedback', methods=['GET'])
def adminViewFeedback():
    try:
        if LoginSession() == 'admin':
            feedbackDAO = FeedbackDAO()
            feedbackVOList = feedbackDAO.viewFeedback()
            loginList = LoginVO.query.filter_by(
                loginId=feedbackVOList[0].feedbackFromLogin).first()
            feedbackVOList[0].feedbackFromLogin = loginList.loginUsername
            print("__________________", feedbackVOList)
            return render_template('admin/viewFeedback.html', feedbackVOList=feedbackVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteFeedback', methods=['GET'])
def adminDeleteFeedback():
    try:
        if LoginSession() == 'admin':
            feedbackVO = FeedbackVO()
            feedbackDAO = FeedbackDAO()
            feedbackId = request.args.get('feedbackId')
            feedbackVO.feedbackId = feedbackId
            feedbackDAO.deleteFeedback(feedbackVO)
            return redirect(url_for('adminViewFeedback'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/reviewFeedback', methods=['GET'])
def adminReviewFeedback():
    try:
        if LoginSession() == 'admin':
            feedbackId = request.args.get('feedbackId')
            feedbackVO = FeedbackVO()
            feedbackDAO = FeedbackDAO()
            feedbackVO.feedbackId = feedbackId
            feedbackVO.feedbackToLogin = session['session_loginId']
            feedbackVO.feedbackReview = "yes"
            feedbackVO.replyDate = datetime.datetime.now().strftime("%d-%m-%Y")
            feedbackVO.replyTime = datetime.datetime.now().strftime("%H:%M:%S")

            feedbackDAO.updateFeedback(feedbackVO)

            return redirect(url_for('adminViewFeedback'))

        else:

            return redirect(url_for('LogoutSession'))

    except Exception as ex:

        print(ex)
