{
  description = "gaze-app: 目線だけで動く仮想カーソルのデスクトップアプリ（顔の動きを補正する / 実カーソルは触らない）";

  inputs = {
    # システム (nixos-26.05) と同じ nixpkgs を使う。store に既にある分は再ビルドされない。
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f system);

      # Python 側の依存。ここに足せばアプリから import できる。
      pythonFor = pkgs: pkgs.python3.withPackages (ps: with ps; [
        numpy
        opencv4
        dlib
        pillow
        tkinter
      ]);

      # アプリ本体を起動するラッパ。`nix run .` か、home-manager から参照して
      # ランチャに載せられるようにしてある。
      gazeApp = pkgs: pkgs.writeShellApplication {
        name = "gaze-app";
        runtimeInputs = [ (pythonFor pkgs) ];
        text = ''
          export GAZE_APP_HOME="''${GAZE_APP_HOME:-$HOME/.cache/gaze-app}"
          # gaze_core.py / gaze_vision.py を import できるようにソース一式を通す
          export PYTHONPATH="${./.}''${PYTHONPATH:+:$PYTHONPATH}"
          exec python3 ${./gaze_app.py} "$@"
        '';
      };
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pythonFor pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [ python ];
            shellHook = ''
              echo "gaze-app 開発環境: python3 gaze_app.py / python3 test_gaze_core.py"
              echo "  カメラは --camera N で選べる。実カーソルは操作しない（仮想カーソルのみ）。"
            '';
          };
        });

      packages = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in
        {
          default = gazeApp pkgs;
          gaze-app = gazeApp pkgs;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${(gazeApp nixpkgs.legacyPackages.${system})}/bin/gaze-app";
        };
      });

      # C / Rust で書いたコアを足すときは、ここに
      #   rustPlatform.buildRustPackage / stdenv.mkDerivation を足して
      #   pythonFor の postPatch か PYTHONPATH で共有ライブラリを渡す。
      # 例:
      #   gazeCore = pkgs.rustPlatform.buildRustPackage { pname = "gaze-core"; ... };
      #   → packages.default の runtimeInputs に加えて LD_LIBRARY_PATH を渡す
    };
}
