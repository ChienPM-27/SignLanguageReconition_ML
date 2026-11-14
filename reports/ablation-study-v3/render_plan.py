"""Render the frozen Vietnamese two-page plan with embedded Unicode fonts."""
import json
from pathlib import Path
from shutil import copy2

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts/ablation-study-v3"
freeze = json.loads((ARTIFACT / "FREEZE_STATUS.json").read_text(encoding="utf-8"))
pdfmetrics.registerFont(TTFont("ArialVN", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("ArialVNBold", "C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFontFamily("ArialVN", normal="ArialVN", bold="ArialVNBold")
navy, accent = colors.HexColor("#17324D"), colors.HexColor("#007F82")
body = ParagraphStyle("body", fontName="ArialVN", fontSize=9.4, leading=14, textColor=navy, spaceAfter=8)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12, spaceAfter=5)
heading = ParagraphStyle("heading", parent=body, fontName="ArialVNBold", fontSize=13, leading=18, spaceBefore=10, spaceAfter=8)
title = ParagraphStyle("title", parent=heading, fontSize=23, leading=29, spaceBefore=0)


def p(text, style=body):
    return Paragraph(text, style)


def table(rows, widths):
    prepared = [[p(str(value), small) for value in row] for row in rows]
    t = Table(prepared, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5F1F1")),
                           ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F8FA")]),
                           ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 8),
                           ("RIGHTPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 7),
                           ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                           ("LINEBELOW", (0,0), (-1,0), 1, accent)]))
    return t


output = ROOT / "output/pdf/ke-hoach-ablation-v3.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=37, bottomMargin=37,
                        title="Kế hoạch ablation V3 - tay, thân, mặt", author="ISLR team")
story = [p("ABLATION STUDY V3", title), p("Tay / thân / mặt - training lại từ đầu", heading),
         p(f"Freeze ID: <b>{freeze['freeze_id']}</b> | 13/11/2026 | Shared chứa code snapshot và toàn bộ configs", small),
         p("<b>Vòng đầu chạy song song: 24 configs / 72 fits.</b> Sau khi có đủ kết quả, kiểm chứng GRU architecture tốt nhất trên các vùng còn lại; tối đa 30 configs / 90 fits. Mỗi config chạy seeds42/43/44."),
         p("Phân công A / B / C", heading),
         table([["Member", "Nhiệm vụ vòng đầu", "Configs / fits"],
                ["A", "GRU G0 (mean_max, một chiều) trên cả 7 tổ hợp vùng", "7 / 21"],
                ["B", "GRU mean pooling, attention pooling và bidirectional; cùng HBF", "3 / 9"],
                ["C", "LSTM + Transformer, mỗi model chạy cùng 7 tổ hợp của A", "14 / 42"],
                ["B - vòng chéo", "Architecture GRU thắng trên HBF: chạy 6 vùng còn lại nếu khác G0", "0 hoặc 6 / 0 hoặc 18"]], [55,350,106]),
         p("Feature ablation và trọng số fusion", heading),
         table([["ID", "Đầu vào model", "Tay / thân / mặt"],
                ["H", "Chỉ bàn tay", "100 / 0 / 0"], ["B", "Chỉ thân trên", "0 / 100 / 0"],
                ["F", "Chỉ mặt", "0 / 0 / 100"], ["HB", "Bàn tay + thân", "77,78 / 22,22 / 0"],
                ["HF", "Bàn tay + mặt", "87,5 / 0 / 12,5"], ["BF", "Thân + mặt", "0 / 66,67 / 33,33"],
                ["HBF", "Bàn tay + thân + mặt (bộ chuẩn)", "70 / 20 / 10"]], [45,285,181]),
         Spacer(1,9), p("Các tỷ lệ là trọng số cộng sau projection128 và L2 normalization. Đây là lựa chọn thiết kế, chưa chứng minh tối ưu và không đảm bảo phần trăm ảnh hưởng tới quyết định model. Frame thiếu một vùng đóng góp0, không chuyển trọng số của nó sang vùng khác.", small),
         PageBreak(), p("CHẠY VÀ NGHIỆM THU", title),
         p("Kaggle: mỗi member một runtime riêng", heading),
         p("<b>1.</b> Giải nén gói bàn giao của member, import notebook vào Kaggle. Attach <b>asl-signs</b> và Dataset chứa <b>Shared V3 ZIP</b>; bật GPU."),
         p("<b>2.</b> Để SHARED_INPUT trống để tự tìm; nếu cần, đặt đường dẫn ZIP hoặc thư mục chứa freeze.json. RAW_INPUT trỏ tới thư mục có train.csv. Giữ MODE=train, RUN_IDS=None rồi <b>Run All</b>."),
         p("<b>3.</b> Sau mỗi config, tải Results ZIP hoặc Save Version giữ Output. Khi session reset, attach ZIP results V3 của mình và điền RESTORE_INPUT để tiếp tục. Configs đã nghiệm thu được bỏ qua."),
         p("Kiểm chứng chéo - có một điểm chờ", heading),
         p("<b>A nhận đủ results B/C</b> rồi dùng MODE=aggregate. Chọn GRU architecture theo HBF: mean Val Macro-F1 cao hơn, std thấp hơn, parameters ít hơn, cuối cùng run ID. Baseline G0/HBF và các GRU của A được dùng lại, không train trùng."),
         p("Nếu G0 thắng, phần GRU đã đủ. Nếu architecture của B thắng, A gửi cross_plan.zip; B restore ZIP vòng đầu, đặt CROSS_PLAN_INPUT rồi Run All để chạy sáu configs bổ sung. A aggregate lại ZIP B mới cùng C."),
         p("Protocol cố định", heading),
         table([["Thành phần", "Giá trị freeze"],
                ["Dữ liệu", "20 glosses; train5.558 / val1.155 / test1.141; signer split15 / 3 / 3"],
                ["Feature", "Tay143 + thân17 (vai, khuỷu tay) + mặt36; cache196; không thêm tip displacement/bend cosines"],
                ["Preprocess", "T32; image/frame; slot tay theo giải phẫu; không trim theo bàn tay"],
                ["Model", "hidden128, layers2, dropout0.3; Transformer heads4, feedforward256"],
                ["Training", "Adam; batch32; lr0.001; weight_decay0.0001; smoothing0.1; tối đa60 epochs"],
                ["Early stopping", "patience15; min_delta0.0001; min_epochs5; checkpoint chọn theo Val Loss"]], [86,425]),
         p("Nghiệm thu và khóa winner", heading),
         p("Đủ ba seed checkpoints, history, metrics, aligned predictions, stage logs, config/source hashes, cache và split snapshots. Aggregate kiểm tra cohort và đối chiếu metrics với predictions. Chỉ chọn winner khi đủ vòng đầu và cross-plan; std là population std, không phải confidence interval.", small),
         p("A dùng MODE=test sau khi aggregate COMPLETED: khóa winner/checkpoint hashes và đánh giá ba checkpoints trên test, không training lại. Kết quả A cũ thuộc V2 vẫn được lưu riêng. Winner chỉ tốt nhất trong các configs V3 đã kiểm chứng.", small)]


def footer(canvas, document):
    canvas.setFont("ArialVN", 8)
    canvas.setFillColor(navy)
    canvas.drawString(42, 22, "ISLR | Ablation V3 | " + freeze["freeze_id"])
    canvas.drawRightString(A4[0]-42, 22, str(document.page))


doc.build(story, onFirstPage=footer, onLaterPages=footer)
copy2(output, ARTIFACT / "KE_HOACH_ABLATION_V3.pdf")
print(output)
