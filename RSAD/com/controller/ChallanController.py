from flask import request, render_template, redirect, url_for
from werkzeug import secure_filename
import os
from RSAD import app
from RSAD.com.dao.VideoDAO import VideoDAO
from RSAD.com.dao.ChallanDAO import ChallanDAO
from RSAD.com.vo.ChallanVO import ChallanVO


# @app.route('/user/UploadChallan', methods=['GET'])
# def userLoadaddChallan():
#    try:
#        videoDAO = VideoDAO()
#        videoVOList = videoDAO.viewVideo()
#        return render_template('user/UploadChallan.html', videoVOList=videoVOList)
#    except Exception as ex:
#        print(ex)


@app.route('/user/insertChallan', methods=['POST'])
def userInsertChallan():
    try:
        ownerName = request.form['ownerName']
        vehNumber = request.form['vehNumber']
        vehType = request.form['vehType']
        imagePath = request.form['image']
        videoPath = request.form['video']

        challanVO = ChallanVO()
        challanDAO = ChallanDAO()

        challanVO.ownerName = ownerName
        challanVO.vehNumber = vehNumber
        challanVO.VehType = vehType
        challanVO.imagePath = imagePath
        challanVO.videoPath = videoPath

        challanDAO.insertChallan(challanVO)
        return render_template('user/login.html')
#        return redirect(url_for('adminViewChallan'))
    except Exception as ex:
        print(ex)


@app.route('/user/viewChallan', methods=['GET'])
def adminViewChallan():
    try:
        challanDAO = ChallanDAO()
        challanVOList = challanDAO.viewChallan()
        print("__________________", challanVOList)
        return render_template('admin/viewChallan.html', challanVOList=challanVOList)
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteChallan', methods=['GET'])
def adminDeleteChallan():
    try:
        challanVO = ChallanVO()

        challanDAO = ChallanDAO()

        challanId = request.args.get('challanId')
        challanpath = request.args.get('imagepath')
        os.remove(imagepath)
        challanVO.challanId = challanId

        challanDAO.deleteChallan(challanVO)

        return redirect(url_for('adminViewChallan'))
    except Exception as ex:
        print(ex)


# @app.route('/admin/editChallan', methods=['GET'])
# def adminEditChallan():
#    try:
#        videoDAO = VideoDAO()
#        videoVOList = videoDAO.viewVideo()
#        challanVO = ChallanVO()
#
#        challanDAO = ChallanDAO()
#
#        challanId = request.args.get('ChallanId')
#
#        challanVO.ChallanId = challanId
#
#        challanVOList = challanDAO.editChallan(challanVO)
#
#        print("=======challanVOList=======", challanVOList)
#
#        print("=======type of challanVOList=======", type(challanVOList))
#
#        return render_template('admin/AddChallan.html', challanVOList=challanVOList,videoVOList=videoVOList)
#    except Exception as ex:
#        print(ex)
#
#
# @app.route('/admin/updateChallan', methods=['POST'])
# def adminUpdateChallan():
#    try:
#        challanId = request.form['ChallanId']
#        challan = request.form['challan']
#        challan_videoId= request.form['challan_videoId']
#
#        challanVO = ChallanVO()
#        challanDAO = ChallanDAO()
#
#        challanVO.ChallanId = challanId
#        challanVO.ChallanName= challanname
#        challanVO.Challan_VideoId= challan_videoId
#
#        challanDAO.updateChallan(challanVO)
#
#        return redirect(url_for('adminViewChallan'))
#    except Exception as ex:
#        print(ex)
