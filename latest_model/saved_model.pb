??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
l
Nadam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_1
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
_output_shapes
: *
dtype0
l
Nadam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/beta_2
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
_output_shapes
: *
dtype0
j
Nadam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameNadam/decay
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
_output_shapes
: *
dtype0
z
Nadam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/learning_rate
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
_output_shapes
: *
dtype0
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Nadam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d/kernel/m
?
)Nadam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
~
Nadam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/conv2d/bias/m
w
'Nadam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Nadam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_1/kernel/m
?
+Nadam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
?
Nadam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_1/bias/m
{
)Nadam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Nadam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameNadam/dense/kernel/m
~
(Nadam/dense/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/m*
_output_shapes
:	?*
dtype0
|
Nadam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/m
u
&Nadam/dense/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense/bias/m*
_output_shapes
:*
dtype0
?
Nadam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/dense_1/kernel/m
?
*Nadam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/m*
_output_shapes

:
*
dtype0
?
Nadam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameNadam/dense_1/bias/m
y
(Nadam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/m*
_output_shapes
:
*
dtype0
?
Nadam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameNadam/conv2d/kernel/v
?
)Nadam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
~
Nadam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameNadam/conv2d/bias/v
w
'Nadam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Nadam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameNadam/conv2d_1/kernel/v
?
+Nadam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
?
Nadam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameNadam/conv2d_1/bias/v
{
)Nadam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Nadam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameNadam/dense/kernel/v
~
(Nadam/dense/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense/kernel/v*
_output_shapes
:	?*
dtype0
|
Nadam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameNadam/dense/bias/v
u
&Nadam/dense/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense/bias/v*
_output_shapes
:*
dtype0
?
Nadam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_nameNadam/dense_1/kernel/v
?
*Nadam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/kernel/v*
_output_shapes

:
*
dtype0
?
Nadam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameNadam/dense_1/bias/v
y
(Nadam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_1/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?D
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?C
value?CB?C B?C
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
R
0	variables
1trainable_variables
2regularization_losses
3	keras_api
R
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
R
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate
Umomentum_cachem?m?&m?'m?<m?=m?Jm?Km?v?v?&v?'v?<v?=v?Jv?Kv?
8
0
1
&2
'3
<4
=5
J6
K7
8
0
1
&2
'3
<4
=5
J6
K7
 
?
	variables
Vnon_trainable_variables
trainable_variables
Wmetrics
regularization_losses

Xlayers
Ylayer_regularization_losses
Zlayer_metrics
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
[non_trainable_variables
trainable_variables
\metrics
regularization_losses

]layers
^layer_regularization_losses
_layer_metrics
 
 
 
?
	variables
`non_trainable_variables
trainable_variables
ametrics
regularization_losses

blayers
clayer_regularization_losses
dlayer_metrics
 
 
 
?
	variables
enon_trainable_variables
trainable_variables
fmetrics
 regularization_losses

glayers
hlayer_regularization_losses
ilayer_metrics
 
 
 
?
"	variables
jnon_trainable_variables
#trainable_variables
kmetrics
$regularization_losses

llayers
mlayer_regularization_losses
nlayer_metrics
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
(	variables
onon_trainable_variables
)trainable_variables
pmetrics
*regularization_losses

qlayers
rlayer_regularization_losses
slayer_metrics
 
 
 
?
,	variables
tnon_trainable_variables
-trainable_variables
umetrics
.regularization_losses

vlayers
wlayer_regularization_losses
xlayer_metrics
 
 
 
?
0	variables
ynon_trainable_variables
1trainable_variables
zmetrics
2regularization_losses

{layers
|layer_regularization_losses
}layer_metrics
 
 
 
?
4	variables
~non_trainable_variables
5trainable_variables
metrics
6regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
8	variables
?non_trainable_variables
9trainable_variables
?metrics
:regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
?
>	variables
?non_trainable_variables
?trainable_variables
?metrics
@regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
B	variables
?non_trainable_variables
Ctrainable_variables
?metrics
Dregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
 
 
 
?
F	variables
?non_trainable_variables
Gtrainable_variables
?metrics
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
L	variables
?non_trainable_variables
Mtrainable_variables
?metrics
Nregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}{
VARIABLE_VALUENadam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUENadam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUENadam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUENadam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUENadam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUENadam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_iptPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_iptconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference_signature_wrapper_40818
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Nadam/conv2d/kernel/m/Read/ReadVariableOp'Nadam/conv2d/bias/m/Read/ReadVariableOp+Nadam/conv2d_1/kernel/m/Read/ReadVariableOp)Nadam/conv2d_1/bias/m/Read/ReadVariableOp(Nadam/dense/kernel/m/Read/ReadVariableOp&Nadam/dense/bias/m/Read/ReadVariableOp*Nadam/dense_1/kernel/m/Read/ReadVariableOp(Nadam/dense_1/bias/m/Read/ReadVariableOp)Nadam/conv2d/kernel/v/Read/ReadVariableOp'Nadam/conv2d/bias/v/Read/ReadVariableOp+Nadam/conv2d_1/kernel/v/Read/ReadVariableOp)Nadam/conv2d_1/bias/v/Read/ReadVariableOp(Nadam/dense/kernel/v/Read/ReadVariableOp&Nadam/dense/bias/v/Read/ReadVariableOp*Nadam/dense_1/kernel/v/Read/ReadVariableOp(Nadam/dense_1/bias/v/Read/ReadVariableOpConst*/
Tin(
&2$	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*'
f"R 
__inference__traced_save_41249
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcounttotal_1count_1Nadam/conv2d/kernel/mNadam/conv2d/bias/mNadam/conv2d_1/kernel/mNadam/conv2d_1/bias/mNadam/dense/kernel/mNadam/dense/bias/mNadam/dense_1/kernel/mNadam/dense_1/bias/mNadam/conv2d/kernel/vNadam/conv2d/bias/vNadam/conv2d_1/kernel/vNadam/conv2d_1/bias/vNadam/dense/kernel/vNadam/dense/bias/vNadam/dense_1/kernel/vNadam/dense_1/bias/v*.
Tin'
%2#*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__traced_restore_41363??
?	
?
*__inference_sequential_layer_call_fn_40938

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_407142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
@__inference_dense_layer_call_and_return_conditional_losses_41054

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_40981

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_1_layer_call_fn_40432

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_404262
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_40604

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_40991

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_40986

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_40520

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_41044

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_405402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_leaky_re_lu_1_layer_call_fn_41006

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_404952
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_40787
ipt
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_407682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
A__inference_conv2d_layer_call_and_return_conditional_losses_40377

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense_1_layer_call_fn_41120

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_406282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_41018

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_40399

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_403932
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_40515

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_40446

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_41039

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_41001

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40426

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_41363
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias!
assignvariableop_8_nadam_iter#
assignvariableop_9_nadam_beta_1$
 assignvariableop_10_nadam_beta_2#
assignvariableop_11_nadam_decay+
'assignvariableop_12_nadam_learning_rate,
(assignvariableop_13_nadam_momentum_cache
assignvariableop_14_total
assignvariableop_15_count
assignvariableop_16_total_1
assignvariableop_17_count_1-
)assignvariableop_18_nadam_conv2d_kernel_m+
'assignvariableop_19_nadam_conv2d_bias_m/
+assignvariableop_20_nadam_conv2d_1_kernel_m-
)assignvariableop_21_nadam_conv2d_1_bias_m,
(assignvariableop_22_nadam_dense_kernel_m*
&assignvariableop_23_nadam_dense_bias_m.
*assignvariableop_24_nadam_dense_1_kernel_m,
(assignvariableop_25_nadam_dense_1_bias_m-
)assignvariableop_26_nadam_conv2d_kernel_v+
'assignvariableop_27_nadam_conv2d_bias_v/
+assignvariableop_28_nadam_conv2d_1_kernel_v-
)assignvariableop_29_nadam_conv2d_1_bias_v,
(assignvariableop_30_nadam_dense_kernel_v*
&assignvariableop_31_nadam_dense_bias_v.
*assignvariableop_32_nadam_dense_1_kernel_v,
(assignvariableop_33_nadam_dense_1_bias_v
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_iterIdentity_8:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_nadam_beta_1Identity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_nadam_beta_2Identity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_nadam_decayIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_nadam_learning_rateIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_nadam_momentum_cacheIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_nadam_conv2d_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_nadam_conv2d_bias_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_nadam_conv2d_1_kernel_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_nadam_conv2d_1_bias_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_nadam_dense_kernel_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp&assignvariableop_23_nadam_dense_bias_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp*assignvariableop_24_nadam_dense_1_kernel_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_nadam_dense_1_bias_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_nadam_conv2d_kernel_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_nadam_conv2d_bias_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_nadam_conv2d_1_kernel_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_nadam_conv2d_1_bias_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_nadam_dense_kernel_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp&assignvariableop_31_nadam_dense_bias_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_nadam_dense_1_kernel_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp(assignvariableop_33_nadam_dense_1_bias_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
?*
?
E__inference_sequential_layer_call_and_return_conditional_losses_40917

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:????????? *
alpha%???>2
leaky_re_lu/LeakyRelu?
dropout/IdentityIdentity#leaky_re_lu/LeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2
dropout/Identity?
max_pooling2d/MaxPoolMaxPooldropout/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_1/LeakyRelu?
dropout_1/IdentityIdentity%leaky_re_lu_1/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2
dropout_1/Identity?
max_pooling2d_1/MaxPoolMaxPooldropout_1/Identity:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_2/LeakyRelu?
dropout_2/IdentityIdentity%leaky_re_lu_2/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_2/Identity?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_2/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????:::::::::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_41023

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_40579

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
z
%__inference_dense_layer_call_fn_41063

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_405582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_dropout_1_layer_call_fn_41033

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?F
?
E__inference_sequential_layer_call_and_return_conditional_losses_40878

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAdd?
leaky_re_lu/LeakyRelu	LeakyReluconv2d/BiasAdd:output:0*/
_output_shapes
:????????? *
alpha%???>2
leaky_re_lu/LeakyRelus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/dropout/Const?
dropout/dropout/MulMul#leaky_re_lu/LeakyRelu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/Mul?
dropout/dropout/ShapeShape#leaky_re_lu/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/dropout/Mul_1?
max_pooling2d/MaxPoolMaxPooldropout/dropout/Mul_1:z:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
leaky_re_lu_1/LeakyRelu	LeakyReluconv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2
leaky_re_lu_1/LeakyReluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul%leaky_re_lu_1/LeakyRelu:activations:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape%leaky_re_lu_1/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul_1?
max_pooling2d_1/MaxPoolMaxPooldropout_1/dropout/Mul_1:z:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
leaky_re_lu_2/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2
leaky_re_lu_2/LeakyReluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMul%leaky_re_lu_2/LeakyRelu:activations:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShape%leaky_re_lu_2/LeakyRelu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout_2/dropout/Mul_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????:::::::::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_40964

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:????????? *
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?2
?
 __inference__wrapped_model_40366
ipt4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity??
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2Dipt/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
sequential/conv2d/BiasAdd?
 sequential/leaky_re_lu/LeakyRelu	LeakyRelu"sequential/conv2d/BiasAdd:output:0*/
_output_shapes
:????????? *
alpha%???>2"
 sequential/leaky_re_lu/LeakyRelu?
sequential/dropout/IdentityIdentity.sequential/leaky_re_lu/LeakyRelu:activations:0*
T0*/
_output_shapes
:????????? 2
sequential/dropout/Identity?
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/dropout/Identity:output:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
sequential/conv2d_1/BiasAdd?
"sequential/leaky_re_lu_1/LeakyRelu	LeakyRelu$sequential/conv2d_1/BiasAdd:output:0*/
_output_shapes
:?????????@*
alpha%???>2$
"sequential/leaky_re_lu_1/LeakyRelu?
sequential/dropout_1/IdentityIdentity0sequential/leaky_re_lu_1/LeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2
sequential/dropout_1/Identity?
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/dropout_1/Identity:output:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
"sequential/leaky_re_lu_2/LeakyRelu	LeakyRelu!sequential/dense/BiasAdd:output:0*'
_output_shapes
:?????????*
alpha%???>2$
"sequential/leaky_re_lu_2/LeakyRelu?
sequential/dropout_2/IdentityIdentity0sequential/leaky_re_lu_2/LeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2
sequential/dropout_2/Identity?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMul&sequential/dropout_2/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/BiasAdd?
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
sequential/dense_1/Softmaxx
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????:::::::::T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
)__inference_dropout_2_layer_call_fn_41095

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_405992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_40714

inputs
conv2d_40684
conv2d_40686
conv2d_1_40692
conv2d_1_40694
dense_40701
dense_40703
dense_1_40708
dense_1_40710
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40684conv2d_40686*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_403772 
conv2d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_404462
leaky_re_lu/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404662!
dropout/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_403932
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_40692conv2d_1_40694*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_404102"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_404952
leaky_re_lu_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405152#
!dropout_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_404262!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_405402
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_40701dense_40703*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_405582
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_405792
leaky_re_lu_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_405992#
!dropout_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_40708dense_1_40710*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_406282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_40540

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
#__inference_signature_wrapper_40818
ipt
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__wrapped_model_403662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_40495

inputs
identityl
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????@*
alpha%???>2
	LeakyRelus
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_41090

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_1_layer_call_fn_40420

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_404102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_41085

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40393

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd~
IdentityIdentityBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_leaky_re_lu_2_layer_call_fn_41073

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_405792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
{
&__inference_conv2d_layer_call_fn_40387

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_403772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_40471

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?/
?
E__inference_sequential_layer_call_and_return_conditional_losses_40768

inputs
conv2d_40738
conv2d_40740
conv2d_1_40746
conv2d_1_40748
dense_40755
dense_40757
dense_1_40762
dense_1_40764
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_40738conv2d_40740*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_403772 
conv2d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_404462
leaky_re_lu/PartitionedCall?
dropout/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404712
dropout/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_403932
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_40746conv2d_1_40748*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_404102"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_404952
leaky_re_lu_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405202
dropout_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_404262!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_405402
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_40755dense_40757*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_405582
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_405792
leaky_re_lu_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_406042
dropout_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_40762dense_1_40764*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_406282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_40466

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_40628

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
*__inference_sequential_layer_call_fn_40733
ipt
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_407142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_leaky_re_lu_layer_call_fn_40969

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_404462
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?/
?
E__inference_sequential_layer_call_and_return_conditional_losses_40678
ipt
conv2d_40648
conv2d_40650
conv2d_1_40656
conv2d_1_40658
dense_40665
dense_40667
dense_1_40672
dense_1_40674
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalliptconv2d_40648conv2d_40650*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_403772 
conv2d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_404462
leaky_re_lu/PartitionedCall?
dropout/PartitionedCallPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404712
dropout/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_403932
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_40656conv2d_1_40658*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_404102"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_404952
leaky_re_lu_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405202
dropout_1/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_404262!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_405402
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_40665dense_40667*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_405582
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_405792
leaky_re_lu_2/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_406042
dropout_2/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_1_40672dense_1_40674*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_406282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_40599

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_40558

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:::P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?Q
?
__inference__traced_save_41249
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_nadam_conv2d_kernel_m_read_readvariableop2
.savev2_nadam_conv2d_bias_m_read_readvariableop6
2savev2_nadam_conv2d_1_kernel_m_read_readvariableop4
0savev2_nadam_conv2d_1_bias_m_read_readvariableop3
/savev2_nadam_dense_kernel_m_read_readvariableop1
-savev2_nadam_dense_bias_m_read_readvariableop5
1savev2_nadam_dense_1_kernel_m_read_readvariableop3
/savev2_nadam_dense_1_bias_m_read_readvariableop4
0savev2_nadam_conv2d_kernel_v_read_readvariableop2
.savev2_nadam_conv2d_bias_v_read_readvariableop6
2savev2_nadam_conv2d_1_kernel_v_read_readvariableop4
0savev2_nadam_conv2d_1_bias_v_read_readvariableop3
/savev2_nadam_dense_kernel_v_read_readvariableop1
-savev2_nadam_dense_bias_v_read_readvariableop5
1savev2_nadam_dense_1_kernel_v_read_readvariableop3
/savev2_nadam_dense_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a8d3287c9d514cee947f19262a75c641/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_nadam_conv2d_kernel_m_read_readvariableop.savev2_nadam_conv2d_bias_m_read_readvariableop2savev2_nadam_conv2d_1_kernel_m_read_readvariableop0savev2_nadam_conv2d_1_bias_m_read_readvariableop/savev2_nadam_dense_kernel_m_read_readvariableop-savev2_nadam_dense_bias_m_read_readvariableop1savev2_nadam_dense_1_kernel_m_read_readvariableop/savev2_nadam_dense_1_bias_m_read_readvariableop0savev2_nadam_conv2d_kernel_v_read_readvariableop.savev2_nadam_conv2d_bias_v_read_readvariableop2savev2_nadam_conv2d_1_kernel_v_read_readvariableop0savev2_nadam_conv2d_1_bias_v_read_readvariableop/savev2_nadam_dense_kernel_v_read_readvariableop-savev2_nadam_dense_bias_v_read_readvariableop1savev2_nadam_dense_1_kernel_v_read_readvariableop/savev2_nadam_dense_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:	?::
:
: : : : : : : : : : : : : @:@:	?::
:
: : : @:@:	?::
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?:  

_output_shapes
::$! 

_output_shapes

:
: "

_output_shapes
:
:#

_output_shapes
: 
?
C
'__inference_dropout_layer_call_fn_40996

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404712
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_41111

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_41068

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:?????????*
alpha%???>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
E__inference_sequential_layer_call_and_return_conditional_losses_40645
ipt
conv2d_40436
conv2d_40438
conv2d_1_40485
conv2d_1_40487
dense_40569
dense_40571
dense_1_40639
dense_1_40641
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCalliptconv2d_40436conv2d_40438*
Tin
2*
Tout
2*/
_output_shapes
:????????? *$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_403772 
conv2d/StatefulPartitionedCall?
leaky_re_lu/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_404462
leaky_re_lu/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall$leaky_re_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_404662!
dropout/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:????????? * 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_403932
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_40485conv2d_1_40487*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_404102"
 conv2d_1/StatefulPartitionedCall?
leaky_re_lu_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_404952
leaky_re_lu_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405152#
!dropout_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_404262!
max_pooling2d_1/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:??????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_405402
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_40569dense_40571*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_405582
dense/StatefulPartitionedCall?
leaky_re_lu_2/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*Q
fLRJ
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_405792
leaky_re_lu_2/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall&leaky_re_lu_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_405992#
!dropout_2/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_1_40639dense_1_40641*
Tin
2*
Tout
2*'
_output_shapes
:?????????
*$
_read_only_resource_inputs
**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_406282!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
/
_output_shapes
:?????????

_user_specified_nameipt:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
)__inference_dropout_1_layer_call_fn_41028

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_405152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_40959

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*'
_output_shapes
:?????????
**
_read_only_resource_inputs

**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_407682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
E
)__inference_dropout_2_layer_call_fn_41100

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_406042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
ipt4
serving_default_ipt:0?????????;
dense_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?G
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-2

layer-9
layer-10
layer-11
layer_with_weights-3
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?D
_tf_keras_sequential?C{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": "accuracy", "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
?
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 25, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1600]}}
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "LeakyReLU", "name": "leaky_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "leaky_re_lu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25]}}
?
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rate
Umomentum_cachem?m?&m?'m?<m?=m?Jm?Km?v?v?&v?'v?<v?=v?Jv?Kv?"
	optimizer
X
0
1
&2
'3
<4
=5
J6
K7"
trackable_list_wrapper
X
0
1
&2
'3
<4
=5
J6
K7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
Vnon_trainable_variables
trainable_variables
Wmetrics
regularization_losses

Xlayers
Ylayer_regularization_losses
Zlayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
[non_trainable_variables
trainable_variables
\metrics
regularization_losses

]layers
^layer_regularization_losses
_layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
`non_trainable_variables
trainable_variables
ametrics
regularization_losses

blayers
clayer_regularization_losses
dlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
enon_trainable_variables
trainable_variables
fmetrics
 regularization_losses

glayers
hlayer_regularization_losses
ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"	variables
jnon_trainable_variables
#trainable_variables
kmetrics
$regularization_losses

llayers
mlayer_regularization_losses
nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(	variables
onon_trainable_variables
)trainable_variables
pmetrics
*regularization_losses

qlayers
rlayer_regularization_losses
slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,	variables
tnon_trainable_variables
-trainable_variables
umetrics
.regularization_losses

vlayers
wlayer_regularization_losses
xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0	variables
ynon_trainable_variables
1trainable_variables
zmetrics
2regularization_losses

{layers
|layer_regularization_losses
}layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
4	variables
~non_trainable_variables
5trainable_variables
metrics
6regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8	variables
?non_trainable_variables
9trainable_variables
?metrics
:regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2dense/kernel
:2
dense/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>	variables
?non_trainable_variables
?trainable_variables
?metrics
@regularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
B	variables
?non_trainable_variables
Ctrainable_variables
?metrics
Dregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
F	variables
?non_trainable_variables
Gtrainable_variables
?metrics
Hregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :
2dense_1/kernel
:
2dense_1/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
L	variables
?non_trainable_variables
Mtrainable_variables
?metrics
Nregularization_losses
?layers
 ?layer_regularization_losses
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: (2Nadam/momentum_cache
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
-:+ 2Nadam/conv2d/kernel/m
: 2Nadam/conv2d/bias/m
/:- @2Nadam/conv2d_1/kernel/m
!:@2Nadam/conv2d_1/bias/m
%:#	?2Nadam/dense/kernel/m
:2Nadam/dense/bias/m
&:$
2Nadam/dense_1/kernel/m
 :
2Nadam/dense_1/bias/m
-:+ 2Nadam/conv2d/kernel/v
: 2Nadam/conv2d/bias/v
/:- @2Nadam/conv2d_1/kernel/v
!:@2Nadam/conv2d_1/bias/v
%:#	?2Nadam/dense/kernel/v
:2Nadam/dense/bias/v
&:$
2Nadam/dense_1/kernel/v
 :
2Nadam/dense_1/bias/v
?2?
*__inference_sequential_layer_call_fn_40733
*__inference_sequential_layer_call_fn_40938
*__inference_sequential_layer_call_fn_40959
*__inference_sequential_layer_call_fn_40787?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_40878
E__inference_sequential_layer_call_and_return_conditional_losses_40917
E__inference_sequential_layer_call_and_return_conditional_losses_40645
E__inference_sequential_layer_call_and_return_conditional_losses_40678?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_40366?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
ipt?????????
?2?
&__inference_conv2d_layer_call_fn_40387?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_40377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
+__inference_leaky_re_lu_layer_call_fn_40969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_40964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_40996
'__inference_dropout_layer_call_fn_40991?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_40986
B__inference_dropout_layer_call_and_return_conditional_losses_40981?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_max_pooling2d_layer_call_fn_40399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40393?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2d_1_layer_call_fn_40420?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
-__inference_leaky_re_lu_1_layer_call_fn_41006?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_41001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_41028
)__inference_dropout_1_layer_call_fn_41033?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_41023
D__inference_dropout_1_layer_call_and_return_conditional_losses_41018?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_max_pooling2d_1_layer_call_fn_40432?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_flatten_layer_call_fn_41044?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_41039?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_41063?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_41054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_leaky_re_lu_2_layer_call_fn_41073?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_41068?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_41095
)__inference_dropout_2_layer_call_fn_41100?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_41090
D__inference_dropout_2_layer_call_and_return_conditional_losses_41085?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_41120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_41111?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.B,
#__inference_signature_wrapper_40818ipt?
 __inference__wrapped_model_40366s&'<=JK4?1
*?'
%?"
ipt?????????
? "1?.
,
dense_1!?
dense_1?????????
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_40410?&'I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
(__inference_conv2d_1_layer_call_fn_40420?&'I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
A__inference_conv2d_layer_call_and_return_conditional_losses_40377?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
&__inference_conv2d_layer_call_fn_40387?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
B__inference_dense_1_layer_call_and_return_conditional_losses_41111\JK/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????

? z
'__inference_dense_1_layer_call_fn_41120OJK/?,
%?"
 ?
inputs?????????
? "??????????
?
@__inference_dense_layer_call_and_return_conditional_losses_41054]<=0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_dense_layer_call_fn_41063P<=0?-
&?#
!?
inputs??????????
? "???????????
D__inference_dropout_1_layer_call_and_return_conditional_losses_41018l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_41023l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
)__inference_dropout_1_layer_call_fn_41028_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
)__inference_dropout_1_layer_call_fn_41033_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
D__inference_dropout_2_layer_call_and_return_conditional_losses_41085\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_41090\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? |
)__inference_dropout_2_layer_call_fn_41095O3?0
)?&
 ?
inputs?????????
p
? "??????????|
)__inference_dropout_2_layer_call_fn_41100O3?0
)?&
 ?
inputs?????????
p 
? "???????????
B__inference_dropout_layer_call_and_return_conditional_losses_40981l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_40986l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
'__inference_dropout_layer_call_fn_40991_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
'__inference_dropout_layer_call_fn_40996_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
B__inference_flatten_layer_call_and_return_conditional_losses_41039a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_41044T7?4
-?*
(?%
inputs?????????@
? "????????????
H__inference_leaky_re_lu_1_layer_call_and_return_conditional_losses_41001h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
-__inference_leaky_re_lu_1_layer_call_fn_41006[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
H__inference_leaky_re_lu_2_layer_call_and_return_conditional_losses_41068X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
-__inference_leaky_re_lu_2_layer_call_fn_41073K/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_leaky_re_lu_layer_call_and_return_conditional_losses_40964h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
+__inference_leaky_re_lu_layer_call_fn_40969[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_40426?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_1_layer_call_fn_40432?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_40393?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
-__inference_max_pooling2d_layer_call_fn_40399?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
E__inference_sequential_layer_call_and_return_conditional_losses_40645o&'<=JK<?9
2?/
%?"
ipt?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_40678o&'<=JK<?9
2?/
%?"
ipt?????????
p 

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_40878r&'<=JK??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_40917r&'<=JK??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
*__inference_sequential_layer_call_fn_40733b&'<=JK<?9
2?/
%?"
ipt?????????
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_40787b&'<=JK<?9
2?/
%?"
ipt?????????
p 

 
? "??????????
?
*__inference_sequential_layer_call_fn_40938e&'<=JK??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
*__inference_sequential_layer_call_fn_40959e&'<=JK??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
#__inference_signature_wrapper_40818z&'<=JK;?8
? 
1?.
,
ipt%?"
ipt?????????"1?.
,
dense_1!?
dense_1?????????
