from flask import request, render_template, redirect, url_for
from RSAD import app
from RSAD.com.dao.AreaDAO import AreaDAO
from RSAD.com.vo.AreaVO import AreaVO
from RSAD.com.controller.LoginController import LoginSession, LogoutSession




@app.route('/admin/insertArea', methods=['POST'])
def adminInsertArea():
    try:
        if LoginSession() == 'admin':
            areaname = request.form['areaName']
            areaPincode= request.form['areaPincode']

            areaVO = AreaVO()
            areaDAO = AreaDAO()

            areaVO.areaName= areaname
            areaVO.areaPincode = areaPincode

            areaDAO.insertArea(areaVO)
    #        return render_template('User/login.html')
            return redirect(url_for('adminViewArea'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/viewArea', methods=['GET'])
def adminViewArea():
    try:
        if LoginSession() == 'admin':
            areaDAO = AreaDAO()
            areaVOList = areaDAO.viewArea()
            print("__________________", areaVOList)
            return render_template('admin/viewArea.html', areaVOList=areaVOList)
        else :
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteArea', methods=['GET'])
def adminDeleteArea():
    try:
        if LoginSession() == 'admin':
            areaVO = AreaVO()

            areaDAO = AreaDAO()

            areaId = request.args.get('areaId')

            areaVO.areaId = areaId

            areaDAO.deleteArea(areaVO)

            return redirect(url_for('adminViewArea'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/editArea', methods=['GET'])
def adminEditArea():
    try:
        if LoginSession() == 'admin':
            print()
            areaVO = AreaVO()

            areaDAO = AreaDAO()

            areaId = request.args.get('areaId')

            areaVO.areaId = areaId

            areaVOList = areaDAO.editArea(areaVO)

            print("=======areaVOList=======", areaVOList)

            print("=======type of areaVOList=======", type(areaVOList))

            return render_template('admin/addArea.html', areaVOList=areaVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/updateArea', methods=['POST'])
def adminUpdateArea():
    try:
        if LoginSession() == 'admin':
            areaId = request.form['areaId']
            areaname = request.form['areaName']
            pincode= request.form['areaPincode']

            areaVO = AreaVO()
            areaDAO = AreaDAO()

            areaVO.areaId = areaId
            areaVO.areaName= areaname
            areaVO.areaPincode= pincode

            areaDAO.updateArea(areaVO)

            return redirect(url_for('adminViewArea'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)
