from django.db import models


class User(models.Model):
    username = models.CharField(max_length=50)
    password = models.CharField(max_length=50)
    email = models.EmailField()

    def __str__(self):
        return self.username

    class Meta:
        verbose_name_plural = 'User'


class Item(models.Model):
    SRV_HUB = models.CharField(max_length=50)
    BRANCH_CODE = models.CharField(max_length=50)
    INV_ITEM_ID = models.CharField(max_length=50)
    USAGE_2013 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2014 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2015 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2016 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2012 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2012 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2012 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2013 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2013 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2013 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2013 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2013 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2013 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2013 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2013 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2013 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2014 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2014 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2014 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2014 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2014 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2014 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2014 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2014 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2014 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2015 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2015 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2015 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2015 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2015 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2015 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2015 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2015 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2015 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2016 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2016 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2016 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2016 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)

    def __str__(self):
        return self.BRANCH_CODE + " " + self.INV_ITEM_ID

    class Meta:
        verbose_name_plural = 'Items'


class Hub17(models.Model):
    SRV_HUB = models.CharField(max_length=50)
    BRANCH_CODE = models.CharField(max_length=50)
    INV_ITEM_ID = models.CharField(max_length=50)
    CATEGORY_ID = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2013 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2014 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2015 = models.DecimalField(max_digits=13, decimal_places=3)
    USAGE_2016 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2012 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2012 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2012 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2013 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2013 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2013 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2013 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2013 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2013 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2013 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2013 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2013 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2013 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2014 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2014 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2014 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2014 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2014 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2014 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2014 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2014 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2014 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2014 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2015 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2015 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2015 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2015 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2015 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2015 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2015 = models.DecimalField(max_digits=13, decimal_places=3)
    OCT2015 = models.DecimalField(max_digits=13, decimal_places=3)
    NOV2015 = models.DecimalField(max_digits=13, decimal_places=3)
    DEC2015 = models.DecimalField(max_digits=13, decimal_places=3)
    JAN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    FEB2016 = models.DecimalField(max_digits=13, decimal_places=3)
    MAR2016 = models.DecimalField(max_digits=13, decimal_places=3)
    APR2016 = models.DecimalField(max_digits=13, decimal_places=3)
    MAY2016 = models.DecimalField(max_digits=13, decimal_places=3)
    JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)

    def __str__(self):
        return self.BRANCH_CODE + " " + self.INV_ITEM_ID

    class Meta:
        verbose_name_plural = 'HUB17_qp~<<'


class Hub17_Result(models.Model):
    SRV_HUB = models.CharField(max_length=50)
    BRANCH_CODE = models.CharField(max_length=50)
    INV_ITEM_ID = models.CharField(max_length=50)
    CATEGORY_ID = models.CharField(max_length=50)
    JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_R2 = models.DecimalField(max_digits=13, decimal_places=3)
    LR_RMSE = models.DecimalField(max_digits=13, decimal_places=3)
    LR_TIME = models.DecimalField(max_digits=13, decimal_places=3)
    AR_JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AR_JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AR_AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AR_SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)
    AR_R2 = models.DecimalField(max_digits=13, decimal_places=3)
    AR_RMSE = models.DecimalField(max_digits=13, decimal_places=3)
    AR_TIME = models.DecimalField(max_digits=13, decimal_places=3)
    LS_JUN2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LS_JUL2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LS_AUG2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LS_SEP2016 = models.DecimalField(max_digits=13, decimal_places=3)
    LS_R2 = models.DecimalField(max_digits=13, decimal_places=3)
    LS_RMSE = models.DecimalField(max_digits=13, decimal_places=3)
    LS_TIME = models.DecimalField(max_digits=13, decimal_places=3)

    def __str__(self):
        return self.BRANCH_CODE + " " + self.INV_ITEM_ID

    class Meta:
        verbose_name_plural = 'HUB17_Result'