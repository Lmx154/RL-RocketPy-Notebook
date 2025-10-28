# This folder is for storing ORK files (Open Rocket Files)

## To start using this, first you will need
 - An ORK file
 - The Open Rocket jar install file <https://openrocket.info/downloads.html?vers=22.02>

## Put both of these files inside the serializer folder.

Change directory into serializer folder

```bash
cd serializer
```

Be neat and create a folder for your rocket

```bash
mkdir rocket
```

Within this directory, use rocket serializer to convert the ork file to a notebook file

```bash
uv run ork2notebook --filepath rocket.ork --output ./rocket
```

or convert to only a parameters.json file

```bash
uv run ork2json --filepath rocket.ork --output ./rocket
```