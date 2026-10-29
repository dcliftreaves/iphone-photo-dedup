#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use serde_json::{json, Value};
use std::{fs, io::{BufRead, BufReader, Write}, path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio}, sync::{Arc, Mutex}, time::{Duration, Instant}};
use tauri::Manager;

struct Engine { child: Child, input: Option<ChildStdin>, output: BufReader<ChildStdout>, sequence: u64 }
impl Engine {
    fn start(app: &tauri::App) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::env::var_os("PHOTO_DEDUP_DATA_DIR").map(PathBuf::from)
            .unwrap_or(app.path().app_data_dir()?);
        fs::create_dir_all(&data)?;
        let root = app.path().resource_dir()?.join("engine");
        let binary = if cfg!(target_os = "windows") { "photo-dedup-engine.exe" } else { "photo-dedup-engine" };
        let packaged = root.join(binary);
        let development = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("engine").join(binary);
        let engine_path = if packaged.exists() { packaged } else { development };
        if !engine_path.exists() { return Err("Bundled photo engine is missing. Run scripts/build_engine.py first.".into()); }
        let log_path = data.join("engine.log");
        if fs::metadata(&log_path).map(|m| m.len() > 1_000_000).unwrap_or(false) {
            let _ = fs::rename(&log_path, data.join("engine.previous.log"));
        }
        let log = fs::OpenOptions::new().create(true).append(true).open(log_path)?;
        let mut command = Command::new(engine_path);
        command.arg("--data-dir").arg(&data).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::from(log));
        #[cfg(target_os = "windows")] {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn()?;
        Ok(Self { input: child.stdin.take(), output: BufReader::new(child.stdout.take().ok_or("Missing worker output")?), child, sequence: 0 })
    }
    fn request(&mut self, method: String, params: Value) -> Result<Value, String> {
        self.sequence += 1;
        let payload = json!({"version":1,"id":self.sequence,"method":method,"params":params});
        let input = self.input.as_mut().ok_or("Engine is shutting down")?;
        writeln!(input, "{}", payload).map_err(|e| e.to_string())?;
        input.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.output.read_line(&mut line).map_err(|e| e.to_string())?;
        if line.is_empty() { return Err("The photo engine stopped. Reopen the app to recover the saved session.".into()); }
        let response: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if response["id"].as_u64() != Some(self.sequence) { return Err("Unexpected engine response".into()); }
        if let Some(error) = response["error"].as_str() { return Err(error.to_string()); }
        Ok(response["result"].clone())
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        // EOF asks the worker to cancel jobs, remove partial exports, and commit caches.
        self.input.take();
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(65) {
            if self.child.try_wait().ok().flatten().is_some() { return; }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill(); let _ = self.child.wait();
    }
}
struct Backend(Arc<Mutex<Engine>>);
#[tauri::command]
async fn rpc(method: String, params: Value, backend: tauri::State<'_, Backend>) -> Result<Value, String> {
    let engine = backend.0.clone();
    tauri::async_runtime::spawn_blocking(move || engine.lock().map_err(|_| "The engine connection was interrupted".to_string())?.request(method, params))
        .await.map_err(|e| e.to_string())?
}
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| {
            if let Some(window) = app.get_webview_window("main") { let _ = window.set_focus(); }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| { let engine = Engine::start(app)?; app.manage(Backend(Arc::new(Mutex::new(engine)))); Ok(()) })
        .invoke_handler(tauri::generate_handler![rpc])
        .run(tauri::generate_context!())
        .expect("Unable to start Photo Dedup");
}
