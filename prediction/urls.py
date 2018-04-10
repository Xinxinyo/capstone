from django.conf.urls import url

from . import views

app_name = 'prediction'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^signin/$', views.signin, name='signin'),
    url(r'^signup/$', views.signup, name='signup'),
    url(r'^signout', views.signout, name='signout'),
    url(r'^authentication', views.authentication, name='authentication'),
    url(r'^predictor_branch$', views.predictor_branch, name='predictor_branch'),
    url(r'^predictor', views.predictor, name='predictor'),
    url(r'^profile', views.profile, name='profile'),
    url(r'^success', views.success, name='success'),
    url(r'^addUser', views.addUser, name='addUser'),
    url(r'^report', views.report, name='report'),
    url(r'^result', views.result, name='result'),
    url(r'^branch_result', views.branch_result, name='branch_result')
]

