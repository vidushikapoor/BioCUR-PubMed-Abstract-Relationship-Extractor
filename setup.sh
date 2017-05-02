#!/bin/bash

sudo apt-get update
sudo apt-get install apache2
sudo apt-get install libapache2-mod-wsgi
sudo apt-get install python-pip
sudo apt-get install python-enchant
sudo pip --no-cache-dir install -r requirements.txt
