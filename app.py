import gradio as gr

def infill_text(input_text):
    inputs = [input_text]
    inputs_masked = [input_text.replace("some_word", "<mask>")]
    outputs, decoded_outputs = rewriter.generate(inputs, inputs_masked, alpha_a=args.alpha_a, alpha_e=args.alpha_e, temperature=args.temperature, verbose=args.verbose, alpha_b=args.alpha_b)
    return decoded_outputs[0]

interface = gr.Interface(fn=infill_text, inputs="text", outputs="text")

# Launch the Gradio interface with share=True for a public link
interface.launch(share=True)
