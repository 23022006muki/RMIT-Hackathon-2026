import { spawn } from 'node:child_process'
import path from 'node:path'

export type Classification = {
  label: 'BENIGN' | 'JAILBREAK' | 'PROMPT_INJECTION'
  modelId: string
  language: string
  truncated: boolean
}

export async function classify(text: string, language: string): Promise<Classification> {
  const checkpoint = process.env.PROTECTOR_CHECKPOINT
  if (!checkpoint) throw new Error('MODEL_UNAVAILABLE')
  const python = process.env.PROTECTOR_PYTHON || path.resolve(process.cwd(), '..', '..', '.venv', 'Scripts', 'python.exe')
  const script = path.resolve(process.cwd(), 'inference', 'classify_once.py')
  return await new Promise<Classification>((resolve, reject) => {
    const child = spawn(/* turbopackIgnore: true */ python, [script, '--checkpoint', checkpoint], {
      cwd: path.resolve(process.cwd(), '..'),
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })
    let output = ''
    let errorOutput = ''
    const timeout = setTimeout(() => child.kill(), 90_000)
    child.stdout.setEncoding('utf8').on('data', (chunk: string) => {
      output += chunk
      if (output.length > 20_000) child.kill()
    })
    child.stderr.setEncoding('utf8').on('data', (chunk: string) => {
      errorOutput += chunk
      if (errorOutput.length > 20_000) child.kill()
    })
    child.on('error', error => { clearTimeout(timeout); reject(error) })
    child.on('close', code => {
      clearTimeout(timeout)
      if (code !== 0) return reject(new Error(`INFERENCE_FAILED: ${errorOutput.slice(-800)}`))
      try {
        const result = JSON.parse(output) as Classification
        if (!['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'].includes(result.label)) throw new Error('Invalid classifier label')
        resolve(result)
      } catch (error) { reject(error) }
    })
    child.stdin.end(JSON.stringify({ text, language }))
  })
}
