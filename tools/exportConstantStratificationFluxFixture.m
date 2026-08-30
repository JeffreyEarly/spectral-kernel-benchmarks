function manifest = exportConstantStratificationFluxFixture(outputDirectory,wvmRepository,options)
% Export one authoritative constant-stratification nonlinear-flux fixture.
%
% The payload contains only the physical configuration, retained mode keys,
% input wave-vortex coefficients, and expected nonlinear flux. skbench must
% independently reconstruct every WVM mode scale and coefficient formula.
arguments (Input)
    outputDirectory (1,1) string {mustBeNonzeroLengthText}
    wvmRepository (1,1) string {mustBeNonzeroLengthText}
    options.Nxyz (1,3) double {mustBeInteger,mustBePositive} = [256 256 129]
    options.Lxyz (1,3) double {mustBePositive} = [15e3 15e3 1300]
    options.N0 (1,1) double {mustBePositive} = 5.2e-3
    options.rotationRate (1,1) double = 7.2921e-5
    options.latitude (1,1) double = 45
    options.g (1,1) double {mustBePositive} = 9.81
    options.elapsedTime (1,1) double = 123.5
    options.seed (1,1) double {mustBeInteger,mustBeNonnegative} = 20020
    options.fixtureId (1,1) string = ""
end
arguments (Output)
    manifest (1,1) struct
end

if options.Nxyz(1) ~= options.Nxyz(2) || options.Lxyz(1) ~= options.Lxyz(2)
    error("SpectralKernelBenchmark:ConstantFluxFixtureRequiresSquareDomain","The constant-stratification fixture requires equal horizontal grid counts and domain lengths.")
end
if mod(options.Nxyz(1),2) ~= 0 || options.Nxyz(3) < 4
    error("SpectralKernelBenchmark:InvalidConstantFluxFixtureSize","The constant-stratification fixture requires an even horizontal grid and Nz >= 4.")
end
if ~isfolder(wvmRepository)
    error("SpectralKernelBenchmark:MissingWVMRepository","The WVM repository does not exist: %s.",wvmRepository)
end
if isfolder(outputDirectory)
    existing = dir(outputDirectory);
    names = string({existing.name});
    existing = existing(~ismember(names,["." ".."]));
    if ~isempty(existing)
        error("SpectralKernelBenchmark:ConstantFluxFixtureExists","The output directory must not contain existing files: %s.",outputDirectory)
    end
else
    mkdir(outputDirectory);
end

toolsFolder = string(fileparts(mfilename("fullpath")));
repositoryRoot = string(fileparts(toolsFolder));
originalPath = path;
originalRng = rng;
stateCleanup = onCleanup(@()restoreState(originalPath,originalRng));
addpath(wvmRepository,toolsFolder);
rng(options.seed,"twister");

matlabTransform = makeTransform(options,"matlab");
compiledTransform = makeTransform(options,"compiled");
if matlabTransform.Nj ~= floor(2*(matlabTransform.Nz-1)/3)
    error("SpectralKernelBenchmark:ConstantFluxFixtureVerticalRetention","WVM retained %d vertical modes; the fixture requires floor(2*(Nz-1)/3) = %d.",matlabTransform.Nj,floor(2*(matlabTransform.Nz-1)/3))
end
if matlabTransform.Nkl ~= compiledTransform.Nkl || matlabTransform.Nj ~= compiledTransform.Nj
    error("SpectralKernelBenchmark:ConstantFluxFixtureBackendShape","The MATLAB and compiled WVM backends disagree on the retained spectral shape.")
end

[Ap,Am,A0] = coefficientFixture(matlabTransform);
elapsedTime = options.elapsedTime;
matlabTransform.Ap = Ap;
matlabTransform.Am = Am;
matlabTransform.A0 = A0;
matlabTransform.t = matlabTransform.t0+elapsedTime;
compiledTransform.Ap = Ap;
compiledTransform.Am = Am;
compiledTransform.A0 = A0;
compiledTransform.t = compiledTransform.t0+elapsedTime;

[matlabFp,matlabFm,matlabF0] = matlabTransform.nonlinearFlux();
[compiledFp,compiledFm,compiledF0] = compiledTransform.nonlinearFlux();
matlabFlux = coefficientBundle(matlabFp,matlabFm,matlabF0);
compiledFlux = coefficientBundle(compiledFp,compiledFm,compiledF0);
[maximumScaleNormalizedError,relativeL2Error] = comparisonError(compiledFlux,matlabFlux);
tolerance = 1e-12;
if maximumScaleNormalizedError > tolerance || relativeL2Error > tolerance
    error("SpectralKernelBenchmark:ConstantFluxFixtureBackendDisagreement","The compiled and MATLAB WVM nonlinear fluxes disagree: maximum scale-normalized %.17g, relative L2 %.17g.",maximumScaleNormalizedError,relativeL2Error)
end

Nj = matlabTransform.Nj;
Nkl = matlabTransform.Nkl;
modeKeys = int32([matlabTransform.kMode_wv(:).';matlabTransform.lMode_wv(:).']);
if ~isequal(modeKeys(:,1),int32([0;0]))
    error("SpectralKernelBenchmark:ConstantFluxFixtureModeOrder","The WVM retained mode order must begin with the zero horizontal mode.")
end
modalState = coefficientBundle(Ap,Am,A0);
payloads = repmat(emptyPayloadRecord(),0,1);
payloads(end+1) = writePayload(outputDirectory,"horizontal-mode-keys.i32le",modeKeys,"int32-le",["coordinate" "mode"],[2 Nkl]);
payloads(end+1) = writeComplexPayload(outputDirectory,"modal-state.c128le",modalState,["j" "coefficient" "mode"],[Nj 3 Nkl]);
payloads(end+1) = writeComplexPayload(outputDirectory,"expected-modal-flux.c128le",compiledFlux,["j" "flux" "mode"],[Nj 3 Nkl]);

wvmSource = sourceRecord(wvmRepository,"JeffreyEarly/wave-vortex-model");
generatorSource = sourceRecord(repositoryRoot,"JeffreyEarly/spectral-kernel-benchmarks");
backend = compiledTransform.computationalBackendMetadata;
isAuthoritative = ~wvmSource.dirtyTree && ~generatorSource.dirtyTree && backend.module.identityValidated && maximumScaleNormalizedError <= tolerance && relativeL2Error <= tolerance;
status = conditional(isAuthoritative,"authoritative-wvm-export","invalid");
if options.fixtureId == ""
    options.fixtureId = sprintf("wvm-constant-stratification-%dx%d-nz%d-f4-seed%d",options.Nxyz(1),options.Nxyz(2),options.Nxyz(3),options.seed);
end

manifest = struct( ...
    "schema","constant-stratification-flux-fixture-v1", ...
    "fixtureId",options.fixtureId, ...
    "status",status, ...
    "authoritative",isAuthoritative, ...
    "createdAtUtc",string(datetime("now","TimeZone","UTC","Format","yyyy-MM-dd'T'HH:mm:ss'Z'")), ...
    "numericType","float64", ...
    "byteOrder","little-endian", ...
    "provenance",wvmSource, ...
    "generator",struct("path","tools/exportConstantStratificationFluxFixture.m","repository",generatorSource.repository,"commit",generatorSource.commit,"tree",generatorSource.tree,"dirtyTree",generatorSource.dirtyTree,"command",generatorCommand(outputDirectory,wvmRepository,options)), ...
    "auditedSources",auditedSources(wvmRepository), ...
    "compiledBackend",struct("provider",backend.provider,"libraries",backend.libraries,"module",backend.module,"contract",backend.contract), ...
    "workload",struct("Nx",matlabTransform.Nx,"Ny",matlabTransform.Ny,"Nz",matlabTransform.Nz,"H",(matlabTransform.Nx/2+1)*matlabTransform.Ny,"Nkl",Nkl,"Nj",Nj,"fields",4,"Lxyz",options.Lxyz), ...
    "physicalConfiguration",struct("N0",options.N0,"rotationRate",options.rotationRate,"latitude",options.latitude,"g",options.g,"isHydrostatic",false,"shouldAntialias",true,"elapsedTime",elapsedTime,"matlabModalNormalizationGravity",9.81), ...
    "retention",struct("horizontalPolicy","radial-two-thirds","horizontalCutoffFraction",2/3,"verticalPolicy","floor(2*(Nz-1)/3)","verticalRetainedFraction",Nj/(matlabTransform.Nz-1)), ...
    "modeOrder",struct("logicalAxes",["k" "l" "j" "coefficient"],"horizontal","WVM radial magnitude then k then l","vertical","j ascending from zero","coefficientNames",["Ap" "Am" "A0"],"fluxNames",["Fp" "Fm" "F0"]), ...
    "normalization",struct("horizontalForward","raw FFT coefficients followed by 1/(Nx*Ny) during modal projection","horizontalInverse","raw inverse FFT with 0.5 type-I inverse factors during coefficient assembly","pointwiseScale",1,"verticalForward","REDFT00/RODFT00 divided by Nz-1 with DCT top endpoint halved and DST endpoints zero"), ...
    "coefficientContract",struct("identity","WVM constant-stratification natural-dimensional-prescaled nonlinear flux","phase","Ap*exp(+i*omega*elapsed), Am*exp(-i*omega*elapsed), A0 unchanged; flux wave contributions returned to reference time","families","U,V use cosine; W,N use sine; target 0/1 derivatives use cosine/cosine/sine; target 2/3 use sine/sine/cosine","specialModes","mode zero uses inertial projection; Kh>0,j=0 is geostrophic; Kh=0,j>0 is mean-density-anomaly"), ...
    "oracle",struct("identity","WVM MATLAB nonlinearFlux cross-checked against the compiled WVTransformConstantStratificationKernel nonlinearFlux","maximumScaleNormalizedError",maximumScaleNormalizedError,"relativeL2Error",relativeL2Error,"maximumScaleNormalizedErrorTolerance",tolerance,"relativeL2ErrorTolerance",tolerance), ...
    "payloads",payloads);

writeText(fullfile(outputDirectory,"manifest.json"),string(jsonencode(manifest,PrettyPrint=true))+newline);
clear stateCleanup
end

function wvt = makeTransform(options,backend)
wvt = WVTransformConstantStratification(options.Lxyz,options.Nxyz,N0=options.N0,rotationRate=options.rotationRate,latitude=options.latitude,g=options.g,shouldAntialias=true,isHydrostatic=false,computationalBackend=backend);
end

function [Ap,Am,A0] = coefficientFixture(wvt)
shape = wvt.spectralMatrixSize;
Ap = complex((2*rand(shape)-1)/8,(2*rand(shape)-1)/8);
Am = complex((2*rand(shape)-1)/8,(2*rand(shape)-1)/8);
A0 = complex((2*rand(shape)-1)/8,(2*rand(shape)-1)/8);
dcMode = find(wvt.kMode_wv == 0 & wvt.lMode_wv == 0,1);
if isempty(dcMode)
    error("SpectralKernelBenchmark:ConstantFluxFixtureMissingDC","The WVM retained mode set does not contain the zero horizontal mode.")
end
Am(:,dcMode) = conj(Ap(:,dcMode));
A0(:,dcMode) = real(A0(:,dcMode));
end

function values = coefficientBundle(first,second,third)
Nj = size(first,1);
Nkl = size(first,2);
values = complex(zeros(Nj,3,Nkl));
values(:,1,:) = reshape(first,Nj,1,Nkl);
values(:,2,:) = reshape(second,Nj,1,Nkl);
values(:,3,:) = reshape(third,Nj,1,Nkl);
end

function [maximumScaleNormalizedError,relativeL2Error] = comparisonError(actual,expected)
difference = actual-expected;
maximumScaleNormalizedError = max(abs(difference),[],"all")/max(1,max(abs(expected),[],"all"));
relativeL2Error = norm(difference(:))/max(1,norm(expected(:)));
end

function payload = writePayload(outputDirectory,fileName,value,elementType,axes,shape)
pathname = fullfile(outputDirectory,fileName);
fileId = fopen(pathname,"w","ieee-le");
if fileId < 0
    error("SpectralKernelBenchmark:ConstantFluxFixtureWriteFailed","Unable to write %s.",pathname)
end
cleanup = onCleanup(@()fclose(fileId));
written = fwrite(fileId,value,"int32");
if written ~= numel(value)
    error("SpectralKernelBenchmark:ConstantFluxFixtureWriteFailed","Expected to write %d elements to %s but wrote %d.",numel(value),pathname,written)
end
clear cleanup
payload = payloadRecord(pathname,fileName,elementType,axes,shape,false);
end

function payload = writeComplexPayload(outputDirectory,fileName,value,axes,shape)
pathname = fullfile(outputDirectory,fileName);
fileId = fopen(pathname,"w","ieee-le");
if fileId < 0
    error("SpectralKernelBenchmark:ConstantFluxFixtureWriteFailed","Unable to write %s.",pathname)
end
cleanup = onCleanup(@()fclose(fileId));
chunkElements = 2^20;
totalWritten = 0;
for firstElement = 1:chunkElements:numel(value)
    lastElement = min(firstElement+chunkElements-1,numel(value));
    chunk = value(firstElement:lastElement);
    interleaved = zeros(2,numel(chunk));
    interleaved(1,:) = real(chunk);
    interleaved(2,:) = imag(chunk);
    totalWritten = totalWritten+fwrite(fileId,interleaved,"double");
end
if totalWritten ~= 2*numel(value)
    error("SpectralKernelBenchmark:ConstantFluxFixtureWriteFailed","Expected to write %d scalar elements to %s but wrote %d.",2*numel(value),pathname,totalWritten)
end
clear cleanup
payload = payloadRecord(pathname,fileName,"complex-float64-interleaved-le",axes,shape,true);
end

function payload = payloadRecord(pathname,fileName,elementType,axes,shape,isComplex)
information = dir(pathname);
strides = ones(size(shape));
for iDimension = 2:numel(shape)
    strides(iDimension) = strides(iDimension-1)*shape(iDimension-1);
end
payload = struct("path",string(fileName),"byteCount",information.bytes,"elementType",string(elementType),"logicalAxes",string(axes),"shape",double(shape),"stridesElements",double(strides),"isComplex",logical(isComplex),"sha256",sha256File(pathname));
end

function payload = emptyPayloadRecord()
payload = struct("path","","byteCount",0,"elementType","","logicalAxes",strings(1,0),"shape",zeros(1,0),"stridesElements",zeros(1,0),"isComplex",false,"sha256","");
end

function source = sourceRecord(repositoryRoot,repositoryIdentity)
[commitStatus,commit] = system("git -C " + shellQuote(repositoryRoot) + " rev-parse HEAD");
[treeStatus,tree] = system("git -C " + shellQuote(repositoryRoot) + " rev-parse HEAD^{tree}");
[dirtyStatus,dirty] = system("git -C " + shellQuote(repositoryRoot) + " status --porcelain --untracked-files=normal");
if commitStatus ~= 0 || treeStatus ~= 0 || dirtyStatus ~= 0
    error("SpectralKernelBenchmark:ConstantFluxFixtureGitProvenance","Unable to read git provenance for %s.",repositoryRoot)
end
source = struct("repository",repositoryIdentity,"commit",strtrim(string(commit)),"tree",strtrim(string(tree)),"dirtyTree",strlength(strtrim(string(dirty))) > 0,"matlabVersion",string(version),"matlabRelease",string(version("-release")),"architecture",string(computer("arch")));
end

function sources = auditedSources(wvmRepository)
paths = ["CompiledKernel/src/WVCoefficientFormulas.hpp" "CompiledKernel/src/WVTransformConstantStratificationKernel.cpp" "CompiledKernel/src/WVKernelTypes.cpp"];
sources = repmat(struct("path","","sha256",""),numel(paths),1);
for iPath = 1:numel(paths)
    sources(iPath).path = paths(iPath);
    sources(iPath).sha256 = sha256File(fullfile(wvmRepository,paths(iPath)));
end
end

function command = generatorCommand(outputDirectory,wvmRepository,options)
command = sprintf("exportConstantStratificationFluxFixture('%s','%s',Nxyz=[%d %d %d],Lxyz=[%.17g %.17g %.17g],N0=%.17g,rotationRate=%.17g,latitude=%.17g,g=%.17g,elapsedTime=%.17g,seed=%d)",outputDirectory,wvmRepository,options.Nxyz,options.Lxyz,options.N0,options.rotationRate,options.latitude,options.g,options.elapsedTime,options.seed);
end

function hash = sha256File(pathname)
fileId = fopen(pathname,"r");
if fileId < 0
    error("SpectralKernelBenchmark:ConstantFluxFixtureHashFailed","Unable to read %s.",pathname)
end
cleanup = onCleanup(@()fclose(fileId));
digest = java.security.MessageDigest.getInstance("SHA-256");
while true
    bytes = fread(fileId,2^24,"*uint8");
    if isempty(bytes)
        break
    end
    digest.update(bytes);
end
hashBytes = typecast(digest.digest(),"uint8");
hash = lower(string(reshape(dec2hex(hashBytes,2).',1,[])));
clear cleanup
end

function writeText(pathname,text)
fileId = fopen(pathname,"w");
if fileId < 0
    error("SpectralKernelBenchmark:ConstantFluxFixtureWriteFailed","Unable to write %s.",pathname)
end
cleanup = onCleanup(@()fclose(fileId));
fprintf(fileId,"%s",text);
clear cleanup
end

function value = shellQuote(value)
value = "'"+replace(string(value),"'","'\''")+"'";
end

function value = conditional(condition,trueValue,falseValue)
if condition
    value = trueValue;
else
    value = falseValue;
end
end

function restoreState(originalPath,originalRng)
path(originalPath);
rng(originalRng);
end
