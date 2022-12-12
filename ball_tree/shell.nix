{pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/49b8ad618e64d9fe9ab686817bfebe047860dcae.tar.gz") {}}:
pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.manim
    pkgs.python3
    pkgs.python3.pkgs.numpy
  ];
}
