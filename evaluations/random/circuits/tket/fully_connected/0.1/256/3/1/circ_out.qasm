OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rz(3*pi/4) q[1];
rz(7*pi/4) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(9*pi/4) q[1];
rz(7*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[1];
rz(11*pi/4) q[2];
rz(3*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(5*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rz(13*pi/4) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rz(5*pi/4) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[1];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(15*pi/4) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[1];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rx(7*pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(11*pi/4) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(9*pi/4) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rx(pi/2) q[1];
rz(15*pi/4) q[2];
rx(7*pi/2) q[0];
rz(3*pi/4) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rz(15*pi/4) q[2];
rz(pi/2) q[0];
rz(7*pi/4) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rz(3*pi/4) q[1];
rz(3*pi/4) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/4) q[0];
rz(5*pi/4) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(15*pi/4) q[0];
rz(pi/2) q[1];
rz(13*pi/4) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(13*pi/4) q[2];
rx(pi/2) q[0];
rz(11*pi/4) q[1];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(3*pi/4) q[2];
rz(13*pi/4) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[1];
rz(13*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[1];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rz(13*pi/4) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(11*pi/4) q[1];
rz(9*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[1];
rz(5*pi/4) q[2];
rz(pi/2) q[0];
rz(7*pi/4) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[1];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[1];
rz(3*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(15*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(7*pi/4) q[1];
rz(7*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(3*pi/4) q[1];
rx(pi/2) q[2];
rz(13*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(5*pi/4) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(11*pi/4) q[1];
rz(7*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(11*pi/4) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(3*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[1];
rz(pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[1];
rx(pi/2) q[2];
rz(pi/4) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rx(pi/2) q[1];
rz(pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/4) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(7*pi/4) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(5*pi/4) q[1];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(11*pi/4) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(9*pi/4) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rz(3*pi/4) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(11*pi/4) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(7*pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[1];
rz(7*pi/4) q[2];
rz(15*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/4) q[0];
rz(3*pi/4) q[1];
rz(7*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(5*pi/4) q[2];
rz(3*pi/4) q[0];
rz(3*pi/4) q[1];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(13*pi/4) q[2];
rz(5*pi/4) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(9*pi/4) q[2];
rz(3*pi/4) q[0];
rz(9*pi/4) q[1];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[1];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[1];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(3*pi/4) q[2];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rz(13*pi/4) q[2];
rz(9*pi/4) q[0];
rz(9*pi/4) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/4) q[0];
rz(3*pi/4) q[1];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[1];
rz(15*pi/4) q[2];
rz(3*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(9*pi/4) q[0];
rz(5*pi/4) q[1];
rz(15*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[1];
rx(7*pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(13*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[1];
rz(3*pi/4) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[1];
rx(pi/2) q[2];
rz(15*pi/4) q[0];
rx(7*pi/2) q[1];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rz(11*pi/4) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rx(pi/2) q[1];
rz(13*pi/4) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(9*pi/4) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[1];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(5*pi/4) q[1];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[1];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[1];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(pi/2) q[1];
rz(9*pi/4) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[1];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi) q[1];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(11*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[2];
rz(pi/4) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(3*pi/4) q[2];
rz(15*pi/4) q[0];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(7*pi/4) q[2];
rz(13*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(5*pi/4) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(7*pi/4) q[2];
rx(7*pi/2) q[0];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(13*pi/4) q[0];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(3*pi/4) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/4) q[2];
rz(15*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[2];
rz(13*pi/4) q[0];
rx(7*pi/2) q[2];
rx(7*pi/2) q[0];
rz(3*pi/4) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[2];
rz(7*pi/4) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[2];
rz(3*pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(7*pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(7*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rz(7*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rz(5*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(13*pi/4) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rz(11*pi/4) q[0];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[2];
rx(pi/2) q[0];
rz(11*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(9*pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rz(9*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(7*pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(13*pi/4) q[2];
rz(15*pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(3*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(15*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(15*pi/4) q[2];
rx(pi/2) q[0];
rx(7*pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(3*pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(13*pi/4) q[2];
rz(pi/2) q[0];
rx(7*pi/2) q[2];
rz(5*pi/4) q[0];
rz(3*pi/4) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rz(13*pi/4) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(5*pi/4) q[0];
rx(pi/2) q[2];
rx(7*pi/2) q[0];
rz(pi/2) q[2];
rz(9*pi/4) q[0];
rx(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/4) q[0];
rz(pi/2) q[2];
rx(7*pi/2) q[0];
rx(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rx(pi/2) q[0];
rz(pi/2) q[2];
rz(pi/2) q[0];
rz(pi/2) q[2];
rz(3*pi/4) q[0];
rx(pi) q[2];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(9*pi/4) q[0];
rx(pi/2) q[0];
rz(9*pi/4) q[0];
rx(7*pi/2) q[0];
rz(13*pi/4) q[0];
rx(pi/2) q[0];
rz(9*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(15*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(13*pi/4) q[0];
rx(pi/2) q[0];
rz(13*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(11*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(5*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(7*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(9*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/4) q[0];
rx(pi/2) q[0];
rz(15*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(13*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(11*pi/4) q[0];
rx(7*pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(13*pi/4) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(11*pi/4) q[0];
rx(7*pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rx(pi) q[0];