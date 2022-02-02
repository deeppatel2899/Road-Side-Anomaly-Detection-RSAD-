from flask import request, render_template, redirect, url_for
from RSAD import app
from RSAD.com.dao.AreaDAO import AreaDAO
from RSAD.com.dao.CrossroadDAO import CrossroadDAO
from RSAD.com.vo.CrossroadVO import CrossroadVO
from RSAD.com.controller.LoginController import LoginSession,LogoutSession


@app.route('/admin/loadCrossroad', methods=['GET'])
def adminLoadCrossroad():
    try:
        if LoginSession() == 'admin':
            areaDAO = AreaDAO()
            areaVOList = areaDAO.viewArea()
            return render_template('admin/addCrossroad.html', areaVOList=areaVOList)
        else:
            return redirect(url_for('LogoutSession'))

    except Exception as ex:
        print(ex)



@app.route('/admin/insertCrossroad', methods=['POST'])
def adminInsertCrossroad():
    try:
        if LoginSession() == 'admin':
            crossroadname = request.form['crossroadName']
            crossroad_areaId= request.form['crossroad_areaId']

            crossroadVO = CrossroadVO()
            crossroadDAO = CrossroadDAO()

            crossroadVO.crossroadName= crossroadname
            crossroadVO.crossroad_areaId= crossroad_areaId

            crossroadDAO.insertCrossroad(crossroadVO)
    #        return render_template('User/login.html')
            return redirect(url_for('adminViewCrossroad'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/viewCrossroad', methods=['GET'])
def adminViewCrossroad():
    try:
        if LoginSession() == 'admin':
            crossroadDAO = CrossroadDAO()
            crossroadVOList = crossroadDAO.viewCrossroad()
            print("__________________", crossroadVOList)
            return render_template('admin/viewCrossroad.html', crossroadVOList=crossroadVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/deleteCrossroad', methods=['GET'])
def adminDeleteCrossroad():
    try:
        if LoginSession() == 'admin':
            crossroadVO = CrossroadVO()

            crossroadDAO = CrossroadDAO()

            crossroadId = request.args.get('crossroadId')

            crossroadVO.crossroadId = crossroadId

            crossroadDAO.deleteCrossroad(crossroadVO)

            return redirect(url_for('adminViewCrossroad'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/editCrossroad', methods=['GET'])
def adminEditCrossroad():
    try:
        if LoginSession() == 'admin':
            areaDAO = AreaDAO()
            areaVOList = areaDAO.viewArea()
            crossroadVO = CrossroadVO()

            crossroadDAO = CrossroadDAO()

            crossroadId = request.args.get('crossroadId')

            crossroadVO.crossroadId = crossroadId

            crossroadVOList = crossroadDAO.editCrossroad(crossroadVO)

            print("=======crossroadVOList=======", crossroadVOList)

            print("=======type of crossroadVOList=======", type(crossroadVOList))

            return render_template('admin/addCrossroad.html', crossroadVOList=crossroadVOList,areaVOList=areaVOList)
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)


@app.route('/admin/updateCrossroad', methods=['POST'])
def adminUpdateCrossroad():
    try:
        if LoginSession() == 'admin':
            crossroadId = request.form['crossroadId']
            crossroadName = request.form['crossroadName']
            crossroad_areaId= request.form['crossroad_areaId']

            crossroadVO = CrossroadVO()
            crossroadDAO = CrossroadDAO()

            crossroadVO.crossroadId = crossroadId
            crossroadVO.crossroadName= crossroadName
            crossroadVO.crossroad_areaId= crossroad_areaId

            crossroadDAO.updateCrossroad(crossroadVO)

            return redirect(url_for('adminViewCrossroad'))
        else:
            return redirect(url_for('LogoutSession'))
    except Exception as ex:
        print(ex)
