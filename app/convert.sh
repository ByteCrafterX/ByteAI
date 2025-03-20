#!/bin/bash
# =============================================================
# test_multiple_conversion_flows.sh
#
# Executa múltiplos testes de conversão de uma imagem utilizando
# cada perfil encontrado, aplicando diferentes combinações de
# parâmetros do ImageMagick.
#
# Aviso: Este script gera muitos arquivos de saída. Teste com
# uma imagem e revise os resultados.
# =============================================================

# Diretório onde estão os arquivos (imagens e perfis)
DIR="/home/marco-alecsander/Scrivania/TEST MARCO"

# Muda para o diretório especificado
cd "$DIR" || { echo "Erro ao acessar o diretório $DIR"; exit 1; }

echo "===================================="
echo "Início dos testes de conversão no diretório: $DIR"
echo "===================================="

# Lista de extensões para imagens e perfis (case-insensitive)
image_extensions="psd psb ifd tif tiff PSD PSB IFD TIF TIFF"
profile_extensions="icc ICC CCH cch"

# Função para iterar sobre os arquivos com determinada extensão (shell nativo não expande lista separada por espaço, então usamos loop)
function for_each_file {
  local ext;
  for ext in $2; do
    for file in *."$ext"; do
      [ -e "$file" ] && echo "$file"
    done
  done
}

# Obtém a lista de imagens (única ou múltipla) – pode ser adaptado se houver mais de uma
images=()
for f in $(for_each_file "$DIR" "$image_extensions"); do
  images+=("$f")
done

if [ ${#images[@]} -eq 0 ]; then
  echo "Nenhuma imagem encontrada com extensões: $image_extensions"
  exit 1
fi

# Obtém a lista de perfis
profiles=()
for p in $(for_each_file "$DIR" "$profile_extensions"); do
  profiles+=("$p")
done

if [ ${#profiles[@]} -eq 0 ]; then
  echo "Nenhum perfil encontrado com extensões: $profile_extensions"
  exit 1
fi

# Lista de testes (fluxos de parâmetros)
# Cada teste é uma função que recebe 3 argumentos:
#   $1 -> imagem de entrada
#   $2 -> perfil a aplicar
#   $3 -> caminho do arquivo de saída
#
# Adicione ou modifique os testes conforme sua necessidade
function teste1 {
  # Fluxo 1: Força entrada em RGB, aplica perfil e converte para sRGB
  convert "$1[0]" -strip -set colorspace RGB -profile "$2" -colorspace sRGB "$3"
}

function teste2 {
  # Fluxo 2: Sem -set colorspace, apenas aplicando o perfil e convertendo para sRGB
  convert "$1[0]" -strip -profile "$2" -colorspace sRGB "$3"
}

function teste3 {
  # Fluxo 3: Aplica o perfil antes de definir o colorspace; pode inverter a ordem
  convert "$1[0]" -strip -profile "$2" -set colorspace RGB -colorspace sRGB "$3"
}

function teste4 {
  # Fluxo 4: Apenas converte para sRGB sem aplicar nenhum perfil (teste de controle)
  convert "$1[0]" -strip -colorspace sRGB "$3"
}

# Array com os nomes dos testes e as funções correspondentes
declare -A testes
testes=(
  ["teste1"]="teste1"
  ["teste2"]="teste2"
  ["teste3"]="teste3"
  ["controle"]="teste4"
)

# Para cada imagem e para cada perfil, executa todos os testes
for image in "${images[@]}"; do
  image_base="${image%.*}"

  echo "---------------------------------------------------"
  echo "Processando imagem: $image"

  for profile in "${profiles[@]}"; do
    profile_base="${profile%.*}"
    echo "  Utilizando perfil: $profile"

    # Executa cada teste
    for test_name in "${!testes[@]}"; do
      # Define o arquivo de saída: image_base + _ + profile_base + _ + test_name.jpg
      output_file="${image_base}_${profile_base}_${test_name}.jpg"

      echo "    Executando $test_name => $output_file"

      # Chama a função do teste passando os parâmetros
      ${testes[$test_name]} "$image" "$profile" "$output_file"

      # Verifica se o arquivo foi criado
      if [ -f "$output_file" ]; then
        echo "      [OK] $output_file criado."
      else
        echo "      [ERRO] Falha na criação de $output_file."
      fi
    done
  done
done

echo "===================================="
echo "Testes de conversão concluídos."
