import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import torch

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network",
                                     epilog="v0.0.1")
    parser.add_argument("input",
                        action="store",
                        help="path to input folder or input pdb file")
    
    parser.add_argument("output",
                        action="store", nargs=argparse.REMAINDER,
                        help="path to output (folder path, npz, or csv)")
    
    parser.add_argument("--csv",
                        "-csv",
                        action="store_true",
                        default=False,
                        help="Writing results to a csv file (Default: False)")
    
    parser.add_argument("--leaveTempFile",
                        "-lt",
                        action="store_true",
                        default=False,
                        help="Leaving temporary files (Default: False)")
    
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="Specifying # of cpus to use for featurization (Default: 1)")
    
    parser.add_argument("--featurize",
                        "-f",
                        action="store_true",
                        default=False,
                        help="Running only the featurization part(Default: False)")
    
    parser.add_argument("--reprocess",
                        "-r", action="store_true",
                        default=False,
                        help="Reprocessing all feature files (Default: False)")
    
    parser.add_argument("--verbose",
                        "-v",
                        action="store_true",
                        default=False,
                        help="Activating verbose flag (Default: False)")
    
    parser.add_argument("--bert",
                        "-bert",
                        action="store_true",
                        default=False,
                        help="Run with bert features. Use extractBert.py to generate them. (Default: False)")
    
    parser.add_argument("--ensemble",
                        "-e", 
                        action="store_true",
                        default=False,
                        help="Running with ensembling of 4 models.  This adds 4x computational time with some overheads (Default: False)")
    args = parser.parse_args()
    
    ################################
    # Checking file availabilities #
    ################################
    csvfilename = "result.csv"
    
    # made outfolder an optional positinal argument. So check manually it's lenght and unpack the string
    if len(args.output)>1:
        print(f"Only one output folder can be specified, but got {args.output}", file=sys.stderr)
        return -1
    
    if len(args.output)==0:
        args.output = ""
    else:
        args.output = args.output[0]

    
    if args.output.endswith(".csv"):
        args.csv = True
        
    if not isdir(args.input):
        print("Input folder does not exist.", file=sys.stderr)
        return -1

    #default is input folder
    if args.output == "":
        args.output = args.input
    else:
        if not args.csv and not isdir(args.output):
            if args.verbose: print("Creating output folder:", args.output)
            os.mkdir(args.output)

        # if csv, do it in place.
        elif args.csv:
            csvfilename = args.output
            args.output = args.input
        
    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    
    if not args.bert:
        modelpath = join(base, "NatComm_FA_distance3D")
    else:
        modelpath = join(base, "NatComm_FA_distance3DBert")
    
    # Eensemble is disabled right now.
    if not isdir(modelpath):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1
        
    ##############################
    # Importing larger libraries #
    ##############################
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import deepAccNet_noPyRosetta as dan
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
        
    #########################
    # Getting samples names #
    #########################
    samples = [i[:-4] for i in os.listdir(args.input) if isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!="."]
    ignored = [i[:-4] for i in os.listdir(args.input) if not(isfile(args.input+"/"+i) and i[-4:] == ".pdb" and i[0]!=".")]
    if args.verbose: 
        print("# samples:", len(samples))
        if len(ignored) > 0:
            print("# files ignored:", len(ignored))

    ##############################
    # Featurization happens here #
    ##############################
    inputs = [join(args.input, s)+".pdb" for s in samples]
    tmpoutputs = [join(args.output, s)+".features.npz" for s in samples]

    if not args.reprocess:
        arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
        already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
        if args.verbose: 
            print("Featurizing", len(arguments), "samples.", len(already_processed), "are already processed.")
    else:
        arguments = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs))]
        already_processed = [(inputs[i], tmpoutputs[i], args.verbose) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
        if args.verbose: 
            print("Featurizing", len(arguments), "samples.", len(already_processed), "are re-processed.")

    if num_process == 1:
        for a in arguments:
            dan.process(a)
    else:
        pool = multiprocessing.Pool(num_process)
        out = pool.map(dan.process, arguments)

    # Exit if only featurization is needed
    if args.featurize:
        return 0

    if args.verbose: print("using", modelpath)

    ###########################
    # Prediction happens here #
    ###########################

    if args.bert:            
        samples = [s for s in samples if isfile(join(args.output, s+".features.npz")) and isfile(join(args.output, "bert_"+s+".npy"))]

    else:
        samples = [s for s in samples if isfile(join(args.output, s+".features.npz"))]

    # Load pytorch model:
    if args.ensemble:
        modelnames = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl"]
    else:
        modelnames = ["best.pkl"]

    result = {}
    for modelname in modelnames:
        model = dan.DeepAccNet_no1D(num_chunks   = 5,
                                    num_channel  = 128,
                                    onebody_size = 0,
                                    twobody_size = 21 if args.bert else 5)
        checkpoint = torch.load(join(modelpath, modelname), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        for s in samples:
            #try:
            with torch.no_grad():
                if args.verbose: print("Predicting for", s) 
                filename = join(args.output, s+".features.npz")
                if args.bert:
                    bertname = join(args.output, "bert_"+s+".npy")
                else:
                    bertname = ""
                (idx, val), (f1d, bert), f2d, dmy = dan.getData(filename, bertpath = bertname)
                f1d = None
                f2d = torch.Tensor(np.expand_dims(f2d.transpose(2,0,1), 0)).to(device)
                idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
                val = torch.Tensor(val).to(device)

                estogram, mask, lddt, dmy = model(idx, val, f1d, f2d)
                t = result.get(s, [])
                t.append(np.mean(lddt.cpu().detach().numpy()))
                result[s] = t

                if not args.csv:
                    if args.ensemble:
                        np.savez_compressed(join(args.output, s+"_"+modelname[:-4]+".npz"),
                                            lddt = lddt.cpu().detach().numpy().astype(np.float16),
                                            estogram = estogram.cpu().detach().numpy().astype(np.float16),
                                            mask = mask.cpu().detach().numpy().astype(np.float16))
                    else:
                        np.savez_compressed(join(args.output, s+".npz"),
                                            lddt = lddt.cpu().detach().numpy().astype(np.float16),
                                            estogram = estogram.cpu().detach().numpy().astype(np.float16),
                                            mask = mask.cpu().detach().numpy().astype(np.float16))
            #except:
            #    print("Failed to predict for", join(args.output, s+"_"+modelname[:-4]+".npz"))

    if not args.csv:

        if args.ensemble:
            dan.merge(samples, args.output, verbose=args.verbose)

        if not args.leaveTempFile:
            dan.clean(samples,
                      args.output,
                      verbose=args.verbose,
                      ensemble=args.ensemble)
    else:
        # Take average of outputs
        csvfile = open(csvfilename, "w")
        csvfile.write("sample\tcb-lddt\n")
        for s in samples:
            line = "%s\t%.4f\n"%(s, np.mean(result[s]))
            csvfile.write(line)
        csvfile.close()
            
if __name__== "__main__":
    main()