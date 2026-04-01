import cadquery as cq

result = (
cq.Workplane()
.workplane(offset=-16)
.circle(31)
.extrude(170)
.faces("<Z")
.workplane()
.circle(23)
.extrude(5)
.faces("<Z")
.workplane()
.polygon(6, 19)
.extrude(12)
)
cq.exporters.export(result, 'GT.stl')