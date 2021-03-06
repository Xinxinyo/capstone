# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-12-04 01:47
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Item',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('SRV_HUB', models.CharField(max_length=50)),
                ('BRANCH_CODE', models.CharField(max_length=50)),
                ('INV_ITEM_ID', models.CharField(max_length=50)),
                ('USAGE_2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('USAGE_2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('USAGE_2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('USAGE_2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('OCT2012', models.DecimalField(decimal_places=3, max_digits=13)),
                ('NOV2012', models.DecimalField(decimal_places=3, max_digits=13)),
                ('DEC2012', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JAN2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('FEB2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAR2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('APR2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAY2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUN2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUL2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('AUG2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('SEP2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('OCT2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('NOV2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('DEC2013', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JAN2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('FEB2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAR2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('APR2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAY2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUN2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUL2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('AUG2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('SEP2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('OCT2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('NOV2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('DEC2014', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JAN2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('FEB2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAR2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('APR2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAY2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUN2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUL2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('AUG2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('SEP2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('OCT2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('NOV2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('DEC2015', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JAN2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('FEB2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAR2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('APR2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('MAY2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUN2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('JUL2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('AUG2016', models.DecimalField(decimal_places=3, max_digits=13)),
                ('SEP2016', models.DecimalField(decimal_places=3, max_digits=13)),
            ],
            options={
                'verbose_name_plural': 'Items',
            },
        ),
        migrations.AlterModelOptions(
            name='user',
            options={'verbose_name_plural': 'User'},
        ),
    ]
