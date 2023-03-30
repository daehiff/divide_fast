OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
cx q[17],q[3];
cx q[19],q[3];
rz(7*pi/4) q[3];
cx q[19],q[3];
cx q[17],q[3];
rz(pi/4) q[19];
rz(3*pi/4) q[2];
cx q[0],q[14];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[6];
cx q[0],q[19];
cx q[0],q[10];
cx q[0],q[4];
rx(5*pi/4) q[0];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[19];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[14];
cx q[5],q[2];
cx q[18],q[2];
cx q[6],q[2];
cx q[19],q[2];
cx q[7],q[2];
rz(pi) q[2];
cx q[7],q[2];
cx q[19],q[2];
cx q[6],q[2];
cx q[18],q[2];
cx q[5],q[2];
cx q[11],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[11],q[0];
cx q[14],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[14],q[8];
cx q[17],q[0];
cx q[11],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[11],q[2];
cx q[21],q[2];
cx q[18],q[2];
cx q[9],q[2];
cx q[3],q[2];
cx q[19],q[2];
cx q[13],q[2];
cx q[16],q[2];
cx q[10],q[2];
cx q[14],q[2];
rz(7*pi/4) q[2];
cx q[14],q[2];
cx q[10],q[2];
cx q[16],q[2];
cx q[13],q[2];
cx q[19],q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[18],q[2];
cx q[21],q[2];
cx q[11],q[2];
cx q[15],q[3];
cx q[22],q[3];
cx q[13],q[3];
cx q[16],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[8],q[3];
rz(pi/2) q[3];
cx q[8],q[3];
cx q[4],q[3];
cx q[7],q[3];
cx q[16],q[3];
cx q[13],q[3];
cx q[22],q[3];
cx q[15],q[3];
cx q[13],q[23];
rx(3*pi/4) q[13];
cx q[13],q[23];
cx q[1],q[9];
cx q[1],q[16];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[23];
cx q[1],q[4];
cx q[1],q[20];
cx q[1],q[11];
cx q[1],q[15];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
cx q[1],q[15];
cx q[1],q[11];
cx q[1],q[20];
cx q[1],q[4];
cx q[1],q[23];
cx q[1],q[7];
cx q[1],q[10];
cx q[1],q[16];
cx q[1],q[9];
cx q[12],q[20];
cx q[12],q[24];
cx q[12],q[16];
rx(pi) q[12];
cx q[12],q[16];
cx q[12],q[24];
cx q[12],q[20];
cx q[9],q[6];
cx q[14],q[6];
cx q[19],q[6];
rz(3*pi/4) q[6];
cx q[19],q[6];
cx q[14],q[6];
cx q[9],q[6];
cx q[24],q[4];
cx q[9],q[4];
rz(3*pi/2) q[4];
cx q[9],q[4];
cx q[24],q[4];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[11];
rx(pi) q[1];
cx q[1],q[11];
cx q[1],q[7];
cx q[1],q[10];
cx q[12],q[5];
cx q[16],q[5];
cx q[20],q[5];
cx q[23],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[23],q[5];
cx q[20],q[5];
cx q[16],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[24],q[3];
rz(pi) q[3];
cx q[24],q[3];
cx q[17],q[3];
cx q[5],q[13];
cx q[5],q[24];
cx q[5],q[19];
cx q[5],q[16];
rx(5*pi/4) q[5];
cx q[5],q[16];
cx q[5],q[19];
cx q[5],q[24];
cx q[5],q[13];
cx q[0],q[21];
cx q[0],q[24];
rx(pi) q[0];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[7];
rx(pi/4) q[0];
cx q[0],q[7];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[21];
cx q[0],q[24];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[18];
cx q[0],q[2];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[2];
cx q[0],q[18];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[1],q[3];
cx q[1],q[13];
cx q[1],q[7];
cx q[1],q[4];
cx q[1],q[21];
cx q[1],q[18];
rx(pi) q[1];
cx q[1],q[18];
cx q[1],q[21];
cx q[1],q[4];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[3];
cx q[19],q[1];
cx q[13],q[1];
cx q[7],q[1];
cx q[14],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[14],q[1];
cx q[7],q[1];
cx q[13],q[1];
cx q[19],q[1];
rz(3*pi/4) q[14];
cx q[2],q[24];
rx(3*pi/2) q[2];
cx q[2],q[24];
cx q[2],q[8];
cx q[2],q[21];
cx q[2],q[15];
cx q[2],q[18];
cx q[2],q[6];
cx q[2],q[20];
cx q[2],q[17];
rx(3*pi/2) q[2];
cx q[2],q[17];
cx q[2],q[20];
cx q[2],q[6];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[21];
cx q[2],q[8];
cx q[22],q[1];
cx q[9],q[1];
cx q[17],q[1];
cx q[20],q[1];
cx q[8],q[1];
cx q[11],q[1];
rz(3*pi/4) q[1];
cx q[11],q[1];
cx q[8],q[1];
cx q[20],q[1];
cx q[17],q[1];
cx q[9],q[1];
cx q[22],q[1];
cx q[7],q[18];
cx q[7],q[12];
cx q[7],q[17];
rx(3*pi/4) q[7];
cx q[7],q[17];
cx q[7],q[12];
cx q[7],q[18];
cx q[3],q[1];
cx q[17],q[1];
cx q[24],q[1];
cx q[11],q[1];
rz(pi) q[1];
cx q[11],q[1];
cx q[24],q[1];
cx q[17],q[1];
cx q[3],q[1];
cx q[2],q[0];
cx q[12],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[12],q[5];
cx q[17],q[5];
cx q[14],q[5];
rz(3*pi/2) q[5];
cx q[14],q[5];
cx q[17],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[19],q[3];
rz(7*pi/4) q[3];
cx q[19],q[3];
cx q[17],q[3];
rz(pi/4) q[19];
rz(3*pi/4) q[2];
cx q[0],q[14];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[6];
cx q[0],q[19];
cx q[0],q[10];
cx q[0],q[4];
rx(5*pi/4) q[0];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[19];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[14];
cx q[5],q[2];
cx q[18],q[2];
cx q[6],q[2];
cx q[19],q[2];
cx q[7],q[2];
rz(pi) q[2];
cx q[7],q[2];
cx q[19],q[2];
cx q[6],q[2];
cx q[18],q[2];
cx q[5],q[2];
cx q[11],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[11],q[0];
cx q[14],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[14],q[8];
cx q[17],q[0];
cx q[11],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[11],q[2];
cx q[21],q[2];
cx q[18],q[2];
cx q[9],q[2];
cx q[3],q[2];
cx q[19],q[2];
cx q[13],q[2];
cx q[16],q[2];
cx q[10],q[2];
cx q[14],q[2];
rz(7*pi/4) q[2];
cx q[14],q[2];
cx q[10],q[2];
cx q[16],q[2];
cx q[13],q[2];
cx q[19],q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[18],q[2];
cx q[21],q[2];
cx q[11],q[2];
cx q[15],q[3];
cx q[22],q[3];
cx q[13],q[3];
cx q[16],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[8],q[3];
rz(pi/2) q[3];
cx q[8],q[3];
cx q[4],q[3];
cx q[7],q[3];
cx q[16],q[3];
cx q[13],q[3];
cx q[22],q[3];
cx q[15],q[3];
cx q[13],q[23];
rx(3*pi/4) q[13];
cx q[13],q[23];
cx q[1],q[9];
cx q[1],q[16];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[23];
cx q[1],q[4];
cx q[1],q[20];
cx q[1],q[11];
cx q[1],q[15];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
cx q[1],q[15];
cx q[1],q[11];
cx q[1],q[20];
cx q[1],q[4];
cx q[1],q[23];
cx q[1],q[7];
cx q[1],q[10];
cx q[1],q[16];
cx q[1],q[9];
cx q[12],q[20];
cx q[12],q[24];
cx q[12],q[16];
rx(pi) q[12];
cx q[12],q[16];
cx q[12],q[24];
cx q[12],q[20];
cx q[9],q[6];
cx q[14],q[6];
cx q[19],q[6];
rz(3*pi/4) q[6];
cx q[19],q[6];
cx q[14],q[6];
cx q[9],q[6];
cx q[24],q[4];
cx q[9],q[4];
rz(3*pi/2) q[4];
cx q[9],q[4];
cx q[24],q[4];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[11];
rx(pi) q[1];
cx q[1],q[11];
cx q[1],q[7];
cx q[1],q[10];
cx q[12],q[5];
cx q[16],q[5];
cx q[20],q[5];
cx q[23],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[23],q[5];
cx q[20],q[5];
cx q[16],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[24],q[3];
rz(pi) q[3];
cx q[24],q[3];
cx q[17],q[3];
cx q[5],q[13];
cx q[5],q[24];
cx q[5],q[19];
cx q[5],q[16];
rx(5*pi/4) q[5];
cx q[5],q[16];
cx q[5],q[19];
cx q[5],q[24];
cx q[5],q[13];
cx q[0],q[21];
cx q[0],q[24];
rx(pi) q[0];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[7];
rx(pi/4) q[0];
cx q[0],q[7];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[21];
cx q[0],q[24];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[18];
cx q[0],q[2];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[2];
cx q[0],q[18];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[1],q[3];
cx q[1],q[13];
cx q[1],q[7];
cx q[1],q[4];
cx q[1],q[21];
cx q[1],q[18];
rx(pi) q[1];
cx q[1],q[18];
cx q[1],q[21];
cx q[1],q[4];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[3];
cx q[19],q[1];
cx q[13],q[1];
cx q[7],q[1];
cx q[14],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[14],q[1];
cx q[7],q[1];
cx q[13],q[1];
cx q[19],q[1];
rz(3*pi/4) q[14];
cx q[2],q[24];
rx(3*pi/2) q[2];
cx q[2],q[24];
cx q[2],q[8];
cx q[2],q[21];
cx q[2],q[15];
cx q[2],q[18];
cx q[2],q[6];
cx q[2],q[20];
cx q[2],q[17];
rx(3*pi/2) q[2];
cx q[2],q[17];
cx q[2],q[20];
cx q[2],q[6];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[21];
cx q[2],q[8];
cx q[22],q[1];
cx q[9],q[1];
cx q[17],q[1];
cx q[20],q[1];
cx q[8],q[1];
cx q[11],q[1];
rz(3*pi/4) q[1];
cx q[11],q[1];
cx q[8],q[1];
cx q[20],q[1];
cx q[17],q[1];
cx q[9],q[1];
cx q[22],q[1];
cx q[7],q[18];
cx q[7],q[12];
cx q[7],q[17];
rx(3*pi/4) q[7];
cx q[7],q[17];
cx q[7],q[12];
cx q[7],q[18];
cx q[3],q[1];
cx q[17],q[1];
cx q[24],q[1];
cx q[11],q[1];
rz(pi) q[1];
cx q[11],q[1];
cx q[24],q[1];
cx q[17],q[1];
cx q[3],q[1];
cx q[2],q[0];
cx q[12],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[12],q[5];
cx q[17],q[5];
cx q[14],q[5];
rz(3*pi/2) q[5];
cx q[14],q[5];
cx q[17],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[19],q[3];
rz(7*pi/4) q[3];
cx q[19],q[3];
cx q[17],q[3];
rz(pi/4) q[19];
rz(3*pi/4) q[2];
cx q[0],q[14];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[6];
cx q[0],q[19];
cx q[0],q[10];
cx q[0],q[4];
rx(5*pi/4) q[0];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[19];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[14];
cx q[5],q[2];
cx q[18],q[2];
cx q[6],q[2];
cx q[19],q[2];
cx q[7],q[2];
rz(pi) q[2];
cx q[7],q[2];
cx q[19],q[2];
cx q[6],q[2];
cx q[18],q[2];
cx q[5],q[2];
cx q[11],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[11],q[0];
cx q[14],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[14],q[8];
cx q[17],q[0];
cx q[11],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[11],q[2];
cx q[21],q[2];
cx q[18],q[2];
cx q[9],q[2];
cx q[3],q[2];
cx q[19],q[2];
cx q[13],q[2];
cx q[16],q[2];
cx q[10],q[2];
cx q[14],q[2];
rz(7*pi/4) q[2];
cx q[14],q[2];
cx q[10],q[2];
cx q[16],q[2];
cx q[13],q[2];
cx q[19],q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[18],q[2];
cx q[21],q[2];
cx q[11],q[2];
cx q[15],q[3];
cx q[22],q[3];
cx q[13],q[3];
cx q[16],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[8],q[3];
rz(pi/2) q[3];
cx q[8],q[3];
cx q[4],q[3];
cx q[7],q[3];
cx q[16],q[3];
cx q[13],q[3];
cx q[22],q[3];
cx q[15],q[3];
cx q[13],q[23];
rx(3*pi/4) q[13];
cx q[13],q[23];
cx q[1],q[9];
cx q[1],q[16];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[23];
cx q[1],q[4];
cx q[1],q[20];
cx q[1],q[11];
cx q[1],q[15];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
cx q[1],q[15];
cx q[1],q[11];
cx q[1],q[20];
cx q[1],q[4];
cx q[1],q[23];
cx q[1],q[7];
cx q[1],q[10];
cx q[1],q[16];
cx q[1],q[9];
cx q[12],q[20];
cx q[12],q[24];
cx q[12],q[16];
rx(pi) q[12];
cx q[12],q[16];
cx q[12],q[24];
cx q[12],q[20];
cx q[9],q[6];
cx q[14],q[6];
cx q[19],q[6];
rz(3*pi/4) q[6];
cx q[19],q[6];
cx q[14],q[6];
cx q[9],q[6];
cx q[24],q[4];
cx q[9],q[4];
rz(3*pi/2) q[4];
cx q[9],q[4];
cx q[24],q[4];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[11];
rx(pi) q[1];
cx q[1],q[11];
cx q[1],q[7];
cx q[1],q[10];
cx q[12],q[5];
cx q[16],q[5];
cx q[20],q[5];
cx q[23],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[23],q[5];
cx q[20],q[5];
cx q[16],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[24],q[3];
rz(pi) q[3];
cx q[24],q[3];
cx q[17],q[3];
cx q[5],q[13];
cx q[5],q[24];
cx q[5],q[19];
cx q[5],q[16];
rx(5*pi/4) q[5];
cx q[5],q[16];
cx q[5],q[19];
cx q[5],q[24];
cx q[5],q[13];
cx q[0],q[21];
cx q[0],q[24];
rx(pi) q[0];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[7];
rx(pi/4) q[0];
cx q[0],q[7];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[21];
cx q[0],q[24];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[18];
cx q[0],q[2];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[2];
cx q[0],q[18];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[1],q[3];
cx q[1],q[13];
cx q[1],q[7];
cx q[1],q[4];
cx q[1],q[21];
cx q[1],q[18];
rx(pi) q[1];
cx q[1],q[18];
cx q[1],q[21];
cx q[1],q[4];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[3];
cx q[19],q[1];
cx q[13],q[1];
cx q[7],q[1];
cx q[14],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[14],q[1];
cx q[7],q[1];
cx q[13],q[1];
cx q[19],q[1];
rz(3*pi/4) q[14];
cx q[2],q[24];
rx(3*pi/2) q[2];
cx q[2],q[24];
cx q[2],q[8];
cx q[2],q[21];
cx q[2],q[15];
cx q[2],q[18];
cx q[2],q[6];
cx q[2],q[20];
cx q[2],q[17];
rx(3*pi/2) q[2];
cx q[2],q[17];
cx q[2],q[20];
cx q[2],q[6];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[21];
cx q[2],q[8];
cx q[22],q[1];
cx q[9],q[1];
cx q[17],q[1];
cx q[20],q[1];
cx q[8],q[1];
cx q[11],q[1];
rz(3*pi/4) q[1];
cx q[11],q[1];
cx q[8],q[1];
cx q[20],q[1];
cx q[17],q[1];
cx q[9],q[1];
cx q[22],q[1];
cx q[7],q[18];
cx q[7],q[12];
cx q[7],q[17];
rx(3*pi/4) q[7];
cx q[7],q[17];
cx q[7],q[12];
cx q[7],q[18];
cx q[3],q[1];
cx q[17],q[1];
cx q[24],q[1];
cx q[11],q[1];
rz(pi) q[1];
cx q[11],q[1];
cx q[24],q[1];
cx q[17],q[1];
cx q[3],q[1];
cx q[2],q[0];
cx q[12],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[12],q[5];
cx q[17],q[5];
cx q[14],q[5];
rz(3*pi/2) q[5];
cx q[14],q[5];
cx q[17],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[19],q[3];
rz(7*pi/4) q[3];
cx q[19],q[3];
cx q[17],q[3];
rz(pi/4) q[19];
rz(3*pi/4) q[2];
cx q[0],q[14];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[6];
cx q[0],q[19];
cx q[0],q[10];
cx q[0],q[4];
rx(5*pi/4) q[0];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[19];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[14];
cx q[5],q[2];
cx q[18],q[2];
cx q[6],q[2];
cx q[19],q[2];
cx q[7],q[2];
rz(pi) q[2];
cx q[7],q[2];
cx q[19],q[2];
cx q[6],q[2];
cx q[18],q[2];
cx q[5],q[2];
cx q[11],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[11],q[0];
cx q[14],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[14],q[8];
cx q[17],q[0];
cx q[11],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[11],q[2];
cx q[21],q[2];
cx q[18],q[2];
cx q[9],q[2];
cx q[3],q[2];
cx q[19],q[2];
cx q[13],q[2];
cx q[16],q[2];
cx q[10],q[2];
cx q[14],q[2];
rz(7*pi/4) q[2];
cx q[14],q[2];
cx q[10],q[2];
cx q[16],q[2];
cx q[13],q[2];
cx q[19],q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[18],q[2];
cx q[21],q[2];
cx q[11],q[2];
cx q[15],q[3];
cx q[22],q[3];
cx q[13],q[3];
cx q[16],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[8],q[3];
rz(pi/2) q[3];
cx q[8],q[3];
cx q[4],q[3];
cx q[7],q[3];
cx q[16],q[3];
cx q[13],q[3];
cx q[22],q[3];
cx q[15],q[3];
cx q[13],q[23];
rx(3*pi/4) q[13];
cx q[13],q[23];
cx q[1],q[9];
cx q[1],q[16];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[23];
cx q[1],q[4];
cx q[1],q[20];
cx q[1],q[11];
cx q[1],q[15];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
cx q[1],q[15];
cx q[1],q[11];
cx q[1],q[20];
cx q[1],q[4];
cx q[1],q[23];
cx q[1],q[7];
cx q[1],q[10];
cx q[1],q[16];
cx q[1],q[9];
cx q[12],q[20];
cx q[12],q[24];
cx q[12],q[16];
rx(pi) q[12];
cx q[12],q[16];
cx q[12],q[24];
cx q[12],q[20];
cx q[9],q[6];
cx q[14],q[6];
cx q[19],q[6];
rz(3*pi/4) q[6];
cx q[19],q[6];
cx q[14],q[6];
cx q[9],q[6];
cx q[24],q[4];
cx q[9],q[4];
rz(3*pi/2) q[4];
cx q[9],q[4];
cx q[24],q[4];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[11];
rx(pi) q[1];
cx q[1],q[11];
cx q[1],q[7];
cx q[1],q[10];
cx q[12],q[5];
cx q[16],q[5];
cx q[20],q[5];
cx q[23],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[23],q[5];
cx q[20],q[5];
cx q[16],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[24],q[3];
rz(pi) q[3];
cx q[24],q[3];
cx q[17],q[3];
cx q[5],q[13];
cx q[5],q[24];
cx q[5],q[19];
cx q[5],q[16];
rx(5*pi/4) q[5];
cx q[5],q[16];
cx q[5],q[19];
cx q[5],q[24];
cx q[5],q[13];
cx q[0],q[21];
cx q[0],q[24];
rx(pi) q[0];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[7];
rx(pi/4) q[0];
cx q[0],q[7];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[21];
cx q[0],q[24];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[18];
cx q[0],q[2];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[2];
cx q[0],q[18];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[1],q[3];
cx q[1],q[13];
cx q[1],q[7];
cx q[1],q[4];
cx q[1],q[21];
cx q[1],q[18];
rx(pi) q[1];
cx q[1],q[18];
cx q[1],q[21];
cx q[1],q[4];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[3];
cx q[19],q[1];
cx q[13],q[1];
cx q[7],q[1];
cx q[14],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[14],q[1];
cx q[7],q[1];
cx q[13],q[1];
cx q[19],q[1];
rz(3*pi/4) q[14];
cx q[2],q[24];
rx(3*pi/2) q[2];
cx q[2],q[24];
cx q[2],q[8];
cx q[2],q[21];
cx q[2],q[15];
cx q[2],q[18];
cx q[2],q[6];
cx q[2],q[20];
cx q[2],q[17];
rx(3*pi/2) q[2];
cx q[2],q[17];
cx q[2],q[20];
cx q[2],q[6];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[21];
cx q[2],q[8];
cx q[22],q[1];
cx q[9],q[1];
cx q[17],q[1];
cx q[20],q[1];
cx q[8],q[1];
cx q[11],q[1];
rz(3*pi/4) q[1];
cx q[11],q[1];
cx q[8],q[1];
cx q[20],q[1];
cx q[17],q[1];
cx q[9],q[1];
cx q[22],q[1];
cx q[7],q[18];
cx q[7],q[12];
cx q[7],q[17];
rx(3*pi/4) q[7];
cx q[7],q[17];
cx q[7],q[12];
cx q[7],q[18];
cx q[3],q[1];
cx q[17],q[1];
cx q[24],q[1];
cx q[11],q[1];
rz(pi) q[1];
cx q[11],q[1];
cx q[24],q[1];
cx q[17],q[1];
cx q[3],q[1];
cx q[2],q[0];
cx q[12],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[12],q[5];
cx q[17],q[5];
cx q[14],q[5];
rz(3*pi/2) q[5];
cx q[14],q[5];
cx q[17],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[19],q[3];
rz(7*pi/4) q[3];
cx q[19],q[3];
cx q[17],q[3];
rz(pi/4) q[19];
rz(3*pi/4) q[2];
cx q[0],q[14];
cx q[0],q[21];
cx q[0],q[15];
cx q[0],q[6];
cx q[0],q[19];
cx q[0],q[10];
cx q[0],q[4];
rx(5*pi/4) q[0];
cx q[0],q[4];
cx q[0],q[10];
cx q[0],q[19];
cx q[0],q[6];
cx q[0],q[15];
cx q[0],q[21];
cx q[0],q[14];
cx q[5],q[2];
cx q[18],q[2];
cx q[6],q[2];
cx q[19],q[2];
cx q[7],q[2];
rz(pi) q[2];
cx q[7],q[2];
cx q[19],q[2];
cx q[6],q[2];
cx q[18],q[2];
cx q[5],q[2];
cx q[11],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[16],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[16],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[11],q[0];
cx q[14],q[8];
cx q[18],q[8];
rz(pi/2) q[8];
cx q[18],q[8];
cx q[14],q[8];
cx q[17],q[0];
cx q[11],q[0];
cx q[8],q[0];
cx q[5],q[0];
cx q[21],q[0];
cx q[15],q[0];
cx q[2],q[0];
cx q[20],q[0];
cx q[7],q[0];
cx q[23],q[0];
rz(pi/4) q[0];
cx q[23],q[0];
cx q[7],q[0];
cx q[20],q[0];
cx q[2],q[0];
cx q[15],q[0];
cx q[21],q[0];
cx q[5],q[0];
cx q[8],q[0];
cx q[11],q[0];
cx q[17],q[0];
cx q[11],q[2];
cx q[21],q[2];
cx q[18],q[2];
cx q[9],q[2];
cx q[3],q[2];
cx q[19],q[2];
cx q[13],q[2];
cx q[16],q[2];
cx q[10],q[2];
cx q[14],q[2];
rz(7*pi/4) q[2];
cx q[14],q[2];
cx q[10],q[2];
cx q[16],q[2];
cx q[13],q[2];
cx q[19],q[2];
cx q[3],q[2];
cx q[9],q[2];
cx q[18],q[2];
cx q[21],q[2];
cx q[11],q[2];
cx q[15],q[3];
cx q[22],q[3];
cx q[13],q[3];
cx q[16],q[3];
cx q[7],q[3];
cx q[4],q[3];
cx q[8],q[3];
rz(pi/2) q[3];
cx q[8],q[3];
cx q[4],q[3];
cx q[7],q[3];
cx q[16],q[3];
cx q[13],q[3];
cx q[22],q[3];
cx q[15],q[3];
cx q[13],q[23];
rx(3*pi/4) q[13];
cx q[13],q[23];
cx q[1],q[9];
cx q[1],q[16];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[23];
cx q[1],q[4];
cx q[1],q[20];
cx q[1],q[11];
cx q[1],q[15];
cx q[1],q[2];
rx(pi/2) q[1];
cx q[1],q[2];
cx q[1],q[15];
cx q[1],q[11];
cx q[1],q[20];
cx q[1],q[4];
cx q[1],q[23];
cx q[1],q[7];
cx q[1],q[10];
cx q[1],q[16];
cx q[1],q[9];
cx q[12],q[20];
cx q[12],q[24];
cx q[12],q[16];
rx(pi) q[12];
cx q[12],q[16];
cx q[12],q[24];
cx q[12],q[20];
cx q[9],q[6];
cx q[14],q[6];
cx q[19],q[6];
rz(3*pi/4) q[6];
cx q[19],q[6];
cx q[14],q[6];
cx q[9],q[6];
cx q[24],q[4];
cx q[9],q[4];
rz(3*pi/2) q[4];
cx q[9],q[4];
cx q[24],q[4];
cx q[1],q[10];
cx q[1],q[7];
cx q[1],q[11];
rx(pi) q[1];
cx q[1],q[11];
cx q[1],q[7];
cx q[1],q[10];
cx q[12],q[5];
cx q[16],q[5];
cx q[20],q[5];
cx q[23],q[5];
cx q[14],q[5];
rz(7*pi/4) q[5];
cx q[14],q[5];
cx q[23],q[5];
cx q[20],q[5];
cx q[16],q[5];
cx q[12],q[5];
cx q[17],q[3];
cx q[24],q[3];
rz(pi) q[3];
cx q[24],q[3];
cx q[17],q[3];
cx q[5],q[13];
cx q[5],q[24];
cx q[5],q[19];
cx q[5],q[16];
rx(5*pi/4) q[5];
cx q[5],q[16];
cx q[5],q[19];
cx q[5],q[24];
cx q[5],q[13];
cx q[0],q[21];
cx q[0],q[24];
rx(pi) q[0];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[17];
cx q[0],q[24];
cx q[0],q[21];
cx q[0],q[9];
cx q[0],q[19];
cx q[0],q[13];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[7];
rx(pi/4) q[0];
cx q[0],q[7];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[13];
cx q[0],q[19];
cx q[0],q[9];
cx q[0],q[21];
cx q[0],q[24];
cx q[0],q[17];
cx q[0],q[11];
cx q[0],q[14];
cx q[0],q[8];
cx q[0],q[18];
cx q[0],q[2];
cx q[0],q[16];
cx q[0],q[20];
cx q[0],q[1];
rx(pi/2) q[0];
cx q[0],q[1];
cx q[0],q[20];
cx q[0],q[16];
cx q[0],q[2];
cx q[0],q[18];
cx q[0],q[8];
cx q[0],q[14];
cx q[0],q[11];
cx q[1],q[3];
cx q[1],q[13];
cx q[1],q[7];
cx q[1],q[4];
cx q[1],q[21];
cx q[1],q[18];
rx(pi) q[1];
cx q[1],q[18];
cx q[1],q[21];
cx q[1],q[4];
cx q[1],q[7];
cx q[1],q[13];
cx q[1],q[3];
cx q[19],q[1];
cx q[13],q[1];
cx q[7],q[1];
cx q[14],q[1];
cx q[24],q[1];
rz(pi/2) q[1];
cx q[24],q[1];
cx q[14],q[1];
cx q[7],q[1];
cx q[13],q[1];
cx q[19],q[1];
rz(3*pi/4) q[14];
cx q[2],q[24];
rx(3*pi/2) q[2];
cx q[2],q[24];
cx q[2],q[8];
cx q[2],q[21];
cx q[2],q[15];
cx q[2],q[18];
cx q[2],q[6];
cx q[2],q[20];
cx q[2],q[17];
rx(3*pi/2) q[2];
cx q[2],q[17];
cx q[2],q[20];
cx q[2],q[6];
cx q[2],q[18];
cx q[2],q[15];
cx q[2],q[21];
cx q[2],q[8];
cx q[22],q[1];
cx q[9],q[1];
cx q[17],q[1];
cx q[20],q[1];
cx q[8],q[1];
cx q[11],q[1];
rz(3*pi/4) q[1];
cx q[11],q[1];
cx q[8],q[1];
cx q[20],q[1];
cx q[17],q[1];
cx q[9],q[1];
cx q[22],q[1];
cx q[7],q[18];
cx q[7],q[12];
cx q[7],q[17];
rx(3*pi/4) q[7];
cx q[7],q[17];
cx q[7],q[12];
cx q[7],q[18];
cx q[3],q[1];
cx q[17],q[1];
cx q[24],q[1];
cx q[11],q[1];
rz(pi) q[1];
cx q[11],q[1];
cx q[24],q[1];
cx q[17],q[1];
cx q[3],q[1];
cx q[2],q[0];
cx q[12],q[0];
cx q[9],q[0];
cx q[3],q[0];
cx q[19],q[0];
cx q[13],q[0];
cx q[4],q[0];
cx q[20],q[0];
cx q[1],q[0];
rz(3*pi/2) q[0];
cx q[1],q[0];
cx q[20],q[0];
cx q[4],q[0];
cx q[13],q[0];
cx q[19],q[0];
cx q[3],q[0];
cx q[9],q[0];
cx q[12],q[0];
cx q[2],q[0];
cx q[12],q[5];
cx q[17],q[5];
cx q[14],q[5];
rz(3*pi/2) q[5];
cx q[14],q[5];
cx q[17],q[5];
cx q[12],q[5];
